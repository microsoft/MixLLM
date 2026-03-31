# FIXME(batchllm): need some code annotations.

from typing import List, Dict


class CSGroupManager():
    def __init__(self, scheduler, block_manager):
        self.scheduler = scheduler
        self.block_manager = block_manager
        self.group_dict = {}
        self.active_group_dict = {}
        self.com2seqid_dict = {}
        self.dist2com_dict = {}
        self.csgroup_request_id_reg = None

    def add_common_request(self, seq_group):
        request_id = seq_group.request_id
        self.csgroup_request_id_reg = request_id
        self.group_dict[request_id] = []

    def add_dist_request(self, seq_group):
        request_id = seq_group.request_id
        if self.csgroup_request_id_reg is not None:
            self.group_dict[self.csgroup_request_id_reg].extend([request_id])
            self.dist2com_dict[request_id] = self.csgroup_request_id_reg
        else:
            # Handle the case where csgroup_request_id_reg is None
            raise ValueError("Common request ID is not registered.")

    def remove_finished_request(self, seq_group):
        request_id = seq_group.request_id
        # common request
        if self.is_csgroup_common(seq_group):
            self.com2seqid_dict[request_id] = {
                'seq_id' : seq_group.get_seqs()[0].seq_id,
                'seq_len' : seq_group.get_seqs()[0].data.get_prompt_len()
            }
            self.active_group_dict[request_id] = self.group_dict.pop(request_id)
            # pop sequence groups from waiting to the dist waiting queue.
            dist_num = len(self.active_group_dict[request_id])
            for i in range(dist_num):
                self.scheduler.dist_ready.appendleft(self.scheduler.dist_waiting.popleft())
        # dist request
        elif self.is_csgroup_dist(seq_group):
            common_request_id = self.dist2com_dict[request_id]
            self.active_group_dict[common_request_id].remove(request_id)
            if self.active_group_dict[common_request_id] == []:
                common_seq_id = self.com2seqid_dict[common_request_id]['seq_id']
                # NOTE(batchllm): There's only one block_manager now.
                self.block_manager.free(None, common_seq_id)
                del self.active_group_dict[common_request_id]
                del self.com2seqid_dict[common_request_id]
                del self.dist2com_dict[request_id]
        # NOTE(xinji1): For the request which is not common nor distinct, we do nothing.
       
    def is_csgroup_common(self, seq_group):
        return seq_group.request_id in self.group_dict or seq_group.request_id in self.active_group_dict
    
    def is_csgroup_dist(self, seq_group):
        return seq_group.request_id in self.dist2com_dict
    
    def is_csgroup(self, seq_group):
        return self.is_csgroup_common(seq_group) or self.is_csgroup_dist(seq_group)
    
    def get_common_request_id(self, seq_group):
        request_id = seq_group.request_id
        if request_id in self.dist2com_dict:
            return self.dist2com_dict[request_id]
        else:
            return request_id


    def get_common_request(self, seq_group):
        request_id = seq_group.request_id
        common_request_id = self.get_common_request_id(seq_group)
        common_block_tables, common_seq_len = [], 0
        if request_id in self.dist2com_dict:
            # dist request
            common_seq_id = self.com2seqid_dict[common_request_id]['seq_id']
            common_block_tables = self.block_manager.get_block_table(None, common_seq_id)
            common_seq_len = self.com2seqid_dict[common_request_id]['seq_len']
        return common_request_id, common_block_tables, common_seq_len