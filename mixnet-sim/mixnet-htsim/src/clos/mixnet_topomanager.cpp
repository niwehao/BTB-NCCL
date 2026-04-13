// -*- c-basic-offset: 4; tab-width: 8; indent-tabs-mode: t -*-
#include "mixnet_topomanager.h"
#include <vector>
#include "string.h"
#include <sstream>
#include <fstream>
#include <strstream>
#include <iostream>
// #include "taskgraph.pb.h"

// #include "main.h"
// #include "queue.h"
// #include "switch.h"
// #include "compositequeue.h"
// #include "prioqueue.h"
// #include "queue_lossless.h"
// #include "queue_lossless_input.h"
// #include "queue_lossless_output.h"
// #include "ecnqueue.h"
// #include "taskgraph_generated.h"
// #include <queue>
// #include <utility>
static std::random_device rd;
static std::mt19937 gen = std::mt19937(rd());
static std::uniform_real_distribution<double> unif(0, 1);
//FIXME: make sure all the src and dst are gpu idx, we need to do the conversion inside the node
#define EDGE(a, b, n) ((a) > (b) ? ((a) * (n) + (b)) : ((b) * (n) + (a)))

extern uint32_t RTT;
extern uint32_t SPEED;
extern ofstream fct_util_out;

All2AllTrafficRecorder::All2AllTrafficRecorder(int layer_num, int region_num, int region_size, TcpRtxTimerScanner * rtx_scanner)
    : rtx_scanner(rtx_scanner), region_size(region_size), region_num(region_num), layer_num(layer_num)
{
  // init the traffic matrix
  traffic_matrix.resize(layer_num);
  for (int i = 0; i < layer_num; i++) {
    traffic_matrix[i].resize(region_num);
    for (int j = 0; j < region_num; j++) {
      // Directly construct the matrix
      traffic_matrix[i][j] = new Matrix2D<double>(region_size, region_size);
    }
  }
}

void All2AllTrafficRecorder::append_traffic_matrix(int layer_id, int region_id, Matrix2D<double> & tm)
{
  // TODO: check the layerid is from 0 or from 1
  assert(layer_id < layer_num);
  assert(region_id < region_num);
  if (!tm.is_symmetric()) {
    tm.transform2symmetric();
  }
  if (traffic_matrix[layer_id][region_id]->get_base() == 0) {
    traffic_matrix[layer_id][region_id]->copy_from(tm);
    traffic_matrix[layer_id][region_id]->update_base(1);
  }
  else {
    // In multi-iteration mode, traffic patterns may evolve.
    // Update the traffic matrix with new data instead of asserting equality.
    if (!traffic_matrix[layer_id][region_id]->check_eq(tm)) {
      traffic_matrix[layer_id][region_id]->copy_from(tm);
    }
  }
}

bool All2AllTrafficRecorder::check_traffic_matrix(int layer_id, int region_id)
{
  // TODO: check the layerid is from 0 or from 1
  assert(layer_id < layer_num);
  assert(region_id < region_num);
  
  if (traffic_matrix[layer_id][region_id]->get_base() == 0) {
    return false;
  }
  else {
    return true;
  }
}



RegionalTopoManager::RegionalTopoManager(Mixnet* topo, All2AllTrafficRecorder* demandrecorder, simtime_picosec refonc_delay, EventList & eventlist, int region_id)
  : EventSource(eventlist, "MixnetRegionalTopoManager"), topo(topo), demandrecorder(demandrecorder), reconf_delay(refonc_delay), eventlist(eventlist), region_id(region_id) {
  
  nnodes = topo->no_of_nodes();
  region_size = topo->region_size;
  region_num = topo->region_num;
  alpha = topo->alpha;
  status = TopoStatus::TOPO_LIVE;

  start_node = region_id * region_size;
  end_node = start_node + region_size;

  // set the regional topo manager for the queues and pipes
  for (int i = start_node; i < end_node; i++)
  {
    for (int j = start_node; j < end_node; j++)
    {
      // if (i == j)
      //   continue;
      Queue *q = topo->queues[i][j];
      ECNQueue *eq = dynamic_cast<ECNQueue *>(q);
      eq->set_regional_topo_manager(this);
      // std::cerr << "setting queue " << eq << std::endl;
      Pipe * p = topo->pipes[i][j];
      p->set_regional_topo_manager(this);
    }
  }
}

void RegionalTopoManager::doNextEvent()
{
  // Reconfiguration event
  if (status == TopoStatus::TOPO_LIVE)
  {
    start_reconf();
    status = TopoStatus::TOPO_RECONF;
  }
  else
  {
    finish_reconf();
    status = TopoStatus::TOPO_LIVE;
    reconfig_end_time = 0;
    // we dont need to trigger this periodically, it will be triggered by computation event
  }
}

void RegionalTopoManager::start_reconf()
{
  // Compute the conn matrix based on traffic matrix at the beginning
  non_empty_queues = 0;
  set_regional_tcp_pause();
  // In multi-collective mode (SimAI), drain wait can deadlock because new flows
  // keep arriving. Force immediate reconfiguration — queued packets will be
  // handled by the new bandwidth assignments.
  _do_reconf();
}

void RegionalTopoManager::do_reconf()
{
  non_empty_queues = 0;
  for (int i = start_node; i < end_node; i++)
  {
    for (int j = start_node; j < end_node; j++)
    {
      if (i == j)
        continue;
      Queue *q = topo->queues[i][j];
      ECNQueue *eq = dynamic_cast<ECNQueue *>(q);
      if (eq->queuesize() > 0)
      {
        // eq->_state_send = LosslessQueue::PAUSE_RECEIVED;
        non_empty_queues += eq->_enqueued.size();
        // std::cerr << i << ", " << j << " qsz " << eq->_enqueued.size() << " bw " << eq->_bitrate << std::endl;
      } 
      else
      {
        // eq->_state_send = LosslessQueue::PAUSED;
      }
      Pipe *p = topo->pipes[i][j];
      if (!p->_inflight.empty())
        non_empty_queues += p->_inflight.size();
    }
  }
  // std::cerr << "non_empty_queues: " << non_empty_queues << std::endl;
  if (non_empty_queues == 0)
  {
    _do_reconf();
  }
}
void RegionalTopoManager::_do_reconf()
{
  update_regional_queue_bandwidth();

  update_regional_route();
  status = TopoStatus::TOPO_RECONF;
  eventlist.sourceIsPendingRel(*this, reconf_delay);
  reconfig_end_time = eventlist.now() + reconf_delay+1;
}

void RegionalTopoManager::finish_reconf()
{
  // resume_lively_queues();
  // pause_no_bw_queues();
  for (int i = start_node; i < end_node; i++)
  {
    for (int j = start_node; j < end_node; j++)
    {
      if (i == j)
        continue;
      Queue *q = topo->queues[i][j];
      ECNQueue *eq = dynamic_cast<ECNQueue *>(q);
      // std::cerr << "queue " << i << ", " << j << " br " << eq->_bitrate << " ps per byte " << eq->_ps_per_byte << " size " << eq->_enqueued.size() << std::endl;
      // Queue may not be empty if new flows arrived during reconf delay
      // assert(eq->_enqueued.empty());
      if (eq->_bitrate > 0)
      {
        eq->_state_send = LosslessQueue::READY;
        if (!eq->_enqueued.empty())
        {
          eq->beginService();
        }
      }
      else
      {
        // No bandwidth — keep paused so packets drain when bandwidth is restored
        eq->_state_send = LosslessQueue::PAUSED;
      }
      // else
      // {
      //   eq->_state_send = LosslessQueue::PAUSED;
      // }
    }
  }
  resume_regional_tcp_flows();
}

void RegionalTopoManager::set_regional_queues_pause_recved()
{
  for (int i = start_node; i < end_node; i++)
  {
    for (int j = start_node; j < end_node; j++)
    {
      if (i == j)
        continue;
      Queue *q = topo->queues[i][j];
      ECNQueue *eq = dynamic_cast<ECNQueue *>(q);
      if (eq->queuesize() > 0)
      {
        // eq->_state_send = LosslessQueue::PAUSE_RECEIVED;
        non_empty_queues += eq->_enqueued.size();
        // std::cerr << i << ", " << j << " qsz " << eq->_enqueued.size() << " bw " << eq->_bitrate << std::endl;
      } 
      else
      {
        // eq->_state_send = LosslessQueue::PAUSED;
      }
      Pipe *p = topo->pipes[i][j];
      if (!p->_inflight.empty())
        non_empty_queues += p->_inflight.size();
    }
  }
  // std
}

void RegionalTopoManager::set_regional_tcp_pause()
{
  /*
    refer to DynFlatScheduler::set_all_tcp_pause()
    1. iter the running tcp flows
    2. if the tcp flow is in this region, set the tcp flow to pause
  */
  list<TcpSrc *>::iterator i = demandrecorder->rtx_scanner->_tcps.begin();
  while (i != demandrecorder->rtx_scanner->_tcps.end())
  {
    TcpSrc *tcpsrc = *i;
    if (is_in_region(tcpsrc))
    {
      assert(tcpsrc->is_all2all);
      tcpsrc->pause_flow_in_region();
    }
    i++;
  }
}

void RegionalTopoManager::update_regional_route()
{
  /*
    In immediate-reconfiguration mode (no drain wait), we do NOT update routes
    for running flows — they continue on their current path to completion.
    Only new flows (created after this reconfig) will use the updated conn matrix
    and get new routes in entry.h's SendFlow().

    This avoids the _nexthop assertion failure caused by changing routes for
    in-flight packets.
  */
}
void RegionalTopoManager::resume_regional_tcp_flows()
{
  /*
    refer to DynFlatScheduler::resume_tcp_flows()
  */
  list<TcpSrc *>::iterator i = demandrecorder->rtx_scanner->_tcps.begin();
  while (i != demandrecorder->rtx_scanner->_tcps.end())
  {
    TcpSrc *tcpsrc = *i;
    if (is_in_region(tcpsrc)) {
      assert(tcpsrc->is_all2all);
      tcpsrc->resume_flow_in_region();
      tcpsrc->resume_flow();
    }
    i++;
  }
}

void RegionalTopoManager::update_regional_queue_bandwidth()
{
  /*
    refer to DynFlatScheduler::update_all_queue_bandwidth()
    1. use the conn matrix computed in start_reconf()
    2. update the queue bandwidth based on conn matrix
  */
  // Use the pre-computed conn matrix from start_reconf()
  reconfig_conn_matrix = regional_topo_reconfig();
  for (int i = 0; i < region_size; i++) {
    for (int j = 0; j < region_size; j++) {
      if (i == j) {
        continue;
      }
      else {
        uint64_t new_bitrate = reconfig_conn_matrix[i][j] * speedFromMbps((uint64_t)SPEED);
        topo->queues[i+start_node][j+start_node]->_bitrate = new_bitrate;
        if (new_bitrate > 0) {
          topo->queues[i+start_node][j+start_node]->_ps_per_byte = (simtime_picosec)((pow(10.0, 12.0) * 8) / new_bitrate);
        } else {
          topo->queues[i+start_node][j+start_node]->_ps_per_byte = 0;
        }
        // Also update reverse direction
        topo->queues[j+start_node][i+start_node]->_bitrate = new_bitrate;
        if (new_bitrate > 0) {
          topo->queues[j+start_node][i+start_node]->_ps_per_byte = (simtime_picosec)((pow(10.0, 12.0) * 8) / new_bitrate);
        } else {
          topo->queues[j+start_node][i+start_node]->_ps_per_byte = 0;
        }
      }
    }
  }
  // direct connect no need to update the path
  // topo->renew_routes_ocs(start_node, end_node); // update regional routes
}

bool RegionalTopoManager::is_in_region(TcpSrc* tcpsrc) const {
  // First check if the flow is finished
  if (tcpsrc->_finished) {
    return false;
  }
  // Then check if it's from all2all task
  // if (!tcpsrc->is_all2all) {
  //   assert(tcpsrc->is_elec);
  //   return false;
  // }
  if (tcpsrc->is_elec) {
    // dont pause elec flows
    return false;
  }
  // Finally check if both source and destination are in this region
  return tcpsrc->_flow_src >= 8*start_node && tcpsrc->_flow_src < 8*end_node && 
         tcpsrc->_flow_dst >= 8*start_node && tcpsrc->_flow_dst < 8*end_node;
}

std::vector<std::vector<int>> RegionalTopoManager::regional_topo_reconfig() {
  Matrix2D<double> traffic_matrix(region_size, region_size);
  traffic_matrix.copy_from(*demandrecorder->traffic_matrix[current_layer_id][region_id]);


  std::vector<std::vector<int>> conn(region_size, std::vector<int>(region_size, 0));
  std::vector<int> avail_conn(region_size, alpha);

  while( std::accumulate(avail_conn.begin(), avail_conn.end(), 0) > 0) {
    std::vector<std::pair<size_t, size_t>> loc = traffic_matrix.get_sorted_indices();
    int is_find = 0;
    int from_idx = 0;
    int to_idx = 0;
    for (auto it = loc.begin(); it != loc.end(); ++it) {
      from_idx = (*it).first;
      to_idx = (*it).second;
      if(from_idx==to_idx){
        //skip when from&to expert locate on same machine
        traffic_matrix.set_elem(from_idx, to_idx, 0);
        continue;
      }
      else{
        is_find=1;
        break;
      }
    }

    assert(is_find==1);
    if(traffic_matrix.get_elem(from_idx, to_idx) == 0) {
      break;
    }
    if(avail_conn[from_idx]>0 && avail_conn[to_idx]>0) {
      conn[from_idx][to_idx] += 1;
      conn[to_idx][from_idx] += 1;
      avail_conn[from_idx]--;
      avail_conn[to_idx]--;
      double new_value = traffic_matrix.get_elem(from_idx,to_idx) / (conn[from_idx][to_idx]*100);
      traffic_matrix.set_elem(from_idx, to_idx, new_value);
      traffic_matrix.set_elem(to_idx, from_idx, new_value);
    }
    if(avail_conn[from_idx]==0) {
      traffic_matrix.set_row(from_idx, 0);
      traffic_matrix.set_col(from_idx, 0);
    }
    if(avail_conn[to_idx]==0){
      traffic_matrix.set_row(to_idx, 0);
      traffic_matrix.set_col(to_idx, 0);
    }
  }
  
  // Debug print removed for performance

  // update the conn of the topo
  for (int i = 0; i < region_size; i++) {
    for (int j = i+1; j < region_size; j++) {
      if(conn[i][j] > 0) {
        topo->conn[i+start_node][j+start_node] = conn[i][j];
        topo->conn[j+start_node][i+start_node] = conn[i][j];
      }
      else {
        topo->conn[i+start_node][j+start_node] = 0;
        topo->conn[j+start_node][i+start_node] = 0;
      }
    }
  }
  return conn;
}

MixnetTopoManager::MixnetTopoManager(Mixnet* topo, All2AllTrafficRecorder* demandrecorder, simtime_picosec refonc_delay, EventList &eventlist)
  : topo(topo), demandrecorder(demandrecorder), reconf_delay(refonc_delay) {
  
  region_num = topo->region_num;
  region_size = topo->region_size;
  nnodes = topo->no_of_nodes();
  alpha = topo->alpha;

  // init the regional topo managers
  for (int i = 0; i < region_num; i++) {
    regional_topo_managers.push_back(new RegionalTopoManager(topo, demandrecorder, reconf_delay, eventlist, i));
  }
}