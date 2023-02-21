import numpy as np
import scipy.special as sps
from typing import Dict, Tuple, List
from heapdict import heapdict
from collections import Counter
import random

class Lattice(object):
    def __init__(self, node_dict, edge_dict):
        self.nodes = node_dict
        self.edges = {node: {} for node in self.nodes} # node -> node -> score
        self.reverse_edges = {node: {} for node in self.nodes}
        for edge_data in edge_dict.values():
            src, tgt, score = edge_data['src'], edge_data['tgt'], edge_data['score']
            assert tgt not in self.edges[src]
            self.edges[src][tgt] = score
            self.reverse_edges[tgt][src] = score
        self.sos, self.eos_list, _ = self._find_start_end(node_dict, edge_dict)
        for eos in self.eos_list: # validate that eos nodes don't have any outgoing edges
            assert len(self.edges[eos]) == 0

    def _find_start_end(self, nodes, edges):
        degree = {}
        out_degree = {}
        for node in nodes.values():
            degree[node['uid']] = 0
            out_degree[node['uid']] = 0
        for edge in edges.values():
            degree[edge['tgt']] += 1
            out_degree[edge['src']] += 1
        key_of_sos = [k for k, v in degree.items() if v == 0]
        key_of_eos = [k for k, v in out_degree.items() if v == 0]
        assert len(key_of_sos) == 1
        return key_of_sos[0], key_of_eos, degree

    def check_cycle(self):
        visited = set()
        def cycle_finder_dfs(curr_node, prev_nodes):
            if curr_node in prev_nodes:
                return True
            if curr_node in visited:
                return False
            visited.add(curr_node)
            prev_nodes.add(curr_node)
            has_cycle = False
            for next_node in self.edges[curr_node]:
                has_cycle = has_cycle or cycle_finder_dfs(next_node, prev_nodes)
            prev_nodes.remove(curr_node)
            return has_cycle
        return cycle_finder_dfs(self.sos, set())

    def _extract_length_dict(self, all_node_length_dict):
        length_dict = {}
        for eos in self.eos_list:
            for length, (length_lprob, length_count) in all_node_length_dict[eos].items():
                if length in length_dict:
                    old_lprob, old_count = length_dict[length]
                    length_dict[length] = (np.logaddexp(old_lprob, length_lprob), old_count + length_count)
                else:
                    length_dict[length] = (length_lprob, length_count)
        keys = list(length_dict.keys())
        #print(keys)
        #print([length_dict[k][1] for k in keys])
        return length_dict

    def get_length_dict_bfs(self):
        '''
        First attempt at DP for mean length calculation.
        Fails if there are cycles in the graph.
        '''
        assert not self.check_cycle()
        node_bag = set()
        node_bag.add(self.sos)
        '''
        node_length_dict = {
          node: {
              length: (score, count)
          }
        }
        '''
        node_length_dict = {node: {} for node in self.nodes}
        node_length_dict[self.sos] = {0: (0, 1)}
        visited = set()

        while len(node_bag) > 0:
            # Only visit a node if all its parents have been visited
            # This only works if there are no cycles in the graph.
            for candidate_node in (self.nodes.keys() - visited):
                all_parents_done = all(parent in visited for parent in self.reverse_edges[candidate_node])
                if all_parents_done:
                    curr_node = candidate_node
                    # node_bag.remove(curr_node)
                    break
            else:
                not_done_parents = {parent for parent in self.reverse_edges[candidate_node] if parent not in visited}
                #print(not_done_parents)
                import pdb; pdb.set_trace()
                raise Exception("No node has all parents done, throwing exception.")
            if curr_node in visited:
                continue
            print(f'{curr_node}: {sum(i for (_, i) in node_length_dict[curr_node].values())}')
            visited.add(curr_node)
            for child_node, edge_score in self.edges[curr_node].items():
                child_node_length_dict = node_length_dict[child_node]
                for length, (length_score, length_count) in node_length_dict[curr_node].items():
                    score_delta = edge_score * length_count + length_score
                    if length + 1 in child_node_length_dict:
                        old_score, old_count = child_node_length_dict[length + 1]
                        new_score = old_score + score_delta
                        new_count = old_count + length_count
                    else:
                        new_score, new_count = score_delta, length_count
                    child_node_length_dict[length + 1] = (new_score, new_count)
                node_bag.add(child_node)
        
        length_dict = self._extract_length_dict(node_length_dict)
        return length_dict, node_length_dict

    def get_length_dict_reverse_dfs(self):
        '''
        For each node v, for all path lengths L, computes the 
        total count & logprob of all paths from SOS to v of length L

        Data stored in dict of following form:
        all_node_length_dict = {
            node: {
                length: (logprob, count)
            }
        }
        '''
        all_node_length_dict: Dict[str, Dict[int, Tuple[float, int]]] = \
            {node: {} for node in self.nodes}
        all_node_length_dict[self.sos] = {0: (0., 1)}

        visited = {self.sos}
        def dfs_helper(node):
            if node in visited:
                return all_node_length_dict[node]
            visited.add(node)

            curr_length_dict = {}
            for parent_node, edge_lprob in self.reverse_edges[node].items():
                parent_length_dict = dfs_helper(parent_node)
                # if len(parent_length_dict) == 0:
                #     # Hit a loop!
                for parent_length, (parent_lprob, parent_count) in parent_length_dict.items():
                    new_length = parent_length + 1
                    # p(h) = p(h[:-1]) * p(h[-1] | h[:-1])
                    added_lprob = parent_lprob + edge_lprob
                    if new_length in curr_length_dict:
                        old_lprob, old_count = curr_length_dict[new_length]
                        new_entry = (np.logaddexp(old_lprob, added_lprob),
                                     old_count + parent_count)
                    else:
                        new_entry = (added_lprob, parent_count)
                    curr_length_dict[new_length] = new_entry
                
            all_node_length_dict[node] = curr_length_dict
            return curr_length_dict

        for eos in self.eos_list:
            dfs_helper(eos)
        
        length_dict = self._extract_length_dict(all_node_length_dict)
        return length_dict, all_node_length_dict

    def _get_node_path_count_dict(self, all_node_length_dict):
        '''
        Returns dict mapping each node to 2-tuple containing
        (total lprob of paths from sos token to that node,
         # of paths from sos token to that node)
        '''
        path_count_dict = {}
        for node, length_data in all_node_length_dict.items():
            #print(length_data)
            all_lprobs, total_count = [-float('inf')], 0
            for length, (lprob, c) in length_data.items():
                all_lprobs.append(lprob)
                total_count += c
            path_count_dict[node] = (sps.logsumexp(all_lprobs), total_count)
        return path_count_dict


    def _get_path_count_bounded_length(self, length_data, target_length, max_deviation):
        '''
        Returns dict mapping each node to 2-tuple containing
        (total lprob of paths from sos token to that node,
         # of paths from sos token to that node)
        '''
        all_lprobs, total_count = [-float('inf')], 0
        for length, (lprob, c) in length_data.items():
            if length <= target_length + max_deviation and length >= target_length - max_deviation:
                all_lprobs.append(lprob)
                total_count += c
        return (sps.logsumexp(all_lprobs), total_count)
    

    def _extract_word_dict(self, all_node_word_dict):
        word_dict = {}
        for eos in self.eos_list:
            for word, (word_lprob, word_count) in all_node_word_dict[eos].items():
                old_lprob, old_count = word_dict.get(word, (-float('inf'),0))
                word_dict[word] = (np.logaddexp(old_lprob, word_lprob), old_count + word_count)
        return word_dict

    def get_1gram_dict(self, all_node_length_dict, target_length=None, allowed_deviation=0):
        '''
        For each node v, for all words w, computes the total count & logprob 
        of all paths from SOS to v that contain w (a unigram)

        all_node_word_dict = {
            node: {
                word: (total probability mass of paths from sos to node that contain word,
                       number of paths from sos to node that contain word)
            }
        }
        '''
        path_count_dict = self._get_node_path_count_dict(all_node_length_dict)

        all_node_word_dict = {node: {} for node in self.nodes}

        visited = {self.sos: {}}
        def dfs_helper(node, length_to_here=0):
            if node == self.sos:
                return all_node_word_dict[node]
            elif target_length is not None:
                if length_to_here >= target_length + allowed_deviation:
                    return all_node_word_dict[node]
                elif node in visited:
                    # want to return if length to here is within deviation of visited length 
                    if True in [(abs(length_to_here - visited_len) <= allowed_deviation) for visited_len in visited[node]]:
                        return all_node_word_dict[node]
                visited[node] = visited.get(node, []) + [length_to_here]
            else:
                if node in visited:
                    return all_node_word_dict[node]
                visited[node] = {}

            curr_word = self.nodes[node]['text']
            curr_word_dict = {} # word -> (total score, count)
            for parent_node, edge_lprob in self.reverse_edges[node].items():
                parent_word_dict = dfs_helper(parent_node, length_to_here=length_to_here+1)
                for word, (parent_lprob, parent_count) in parent_word_dict.items():
                    if parent_count == 0:
                        continue
                    if word != curr_word:
                        added_lprob = parent_lprob + edge_lprob
                        if word in curr_word_dict:
                            old_lprob, old_count = curr_word_dict[word]
                            curr_word_dict[word] = (np.logaddexp(old_lprob, added_lprob),
                                                    old_count + parent_count)
                        else:
                            curr_word_dict[word] = (added_lprob, parent_count)

            # The number of paths that contain the current word is equal
            # to the number of paths that reach the current node.
            if target_length is not None:
                curr_word_dict[curr_word] = self._get_path_count_bounded_length(all_node_length_dict[node], target_length - (length_to_here + 1), allowed_deviation)
            else:
                curr_word_dict[curr_word] = path_count_dict[node]
            
            all_node_word_dict[node] = curr_word_dict
            return curr_word_dict

        for eos in self.eos_list:
            dfs_helper(eos)
        
        word_dict = self._extract_word_dict(all_node_word_dict)
        return word_dict, all_node_word_dict

    def get_1gram_dict_count_aware(self, all_node_length_dict, target_length=None, allowed_deviation=0):
        '''
        Same as get_1gram_dict, but with count-aware adjustment
        to counts.

        Practically, we treat the first and second occurrences of a word
        in a sentence as different words by combining each word with its 
        occurrence number in the sequence.

        all_node_word_dict = {
            node: {
                word: (total probability mass of paths from sos to node that contain word,
                       number of paths from sos to node that contain word)
            }
        }
        '''
        path_count_dict = self._get_node_path_count_dict(all_node_length_dict)

        all_node_word_dict = {node: {} for node in self.nodes}

        visited = {self.sos}
        def dfs_helper(node):
            if node in visited:
                return all_node_word_dict[node]
            visited.add(node)

            curr_word = self.nodes[node]['text']

            curr_word_dict = {} # word -> (total score, count)
            for parent_node, edge_lprob in self.reverse_edges[node].items():
                parent_word_dict = dfs_helper(parent_node)
                for word, (parent_lprob, parent_count) in parent_word_dict.items():
                    added_lprob = parent_lprob + edge_lprob
                    if word in curr_word_dict:
                        old_lprob, old_count = curr_word_dict[word]
                        curr_word_dict[word] = (np.logaddexp(old_lprob, added_lprob),
                                                old_count + parent_count)
                    else:
                        curr_word_dict[word] = (added_lprob, parent_count)

            total_lprob, total_paths = path_count_dict[node]
            if total_lprob != float('inf'):
                max_count = 1
                while (curr_word, max_count) in curr_word_dict:
                    max_count += 1
                other_count_paths = 0
                other_count_scores = [total_lprob]
                for count in range(max_count, 1, -1): # count \in [max_count, ..., 2]
                    curr_word_dict[(curr_word, count)] = curr_word_dict[(curr_word, count-1)]
                    other_count_scores.append(curr_word_dict[(curr_word, count-1)][0])
                    other_count_paths += curr_word_dict[(curr_word, count-1)][1]
                scaling = -np.ones(len(other_count_scores)) # want to subtract lprobs
                scaling[0] = 1
                remaining_lprob = sps.logsumexp(other_count_scores, b=scaling)
                if np.isnan(remaining_lprob):
                    remaining_lprob = -float('inf')
                curr_word_dict[(curr_word, 1)] = (
                    remaining_lprob,
                    total_paths - other_count_paths
                )

            all_node_word_dict[node] = curr_word_dict
            return curr_word_dict

        for eos in self.eos_list:
            dfs_helper(eos)
        
        word_dict = self._extract_word_dict(all_node_word_dict)
        return word_dict, all_node_word_dict

    def get_2gram_dict(self, all_node_length_dict, target_length=None, allowed_deviation=0):
        '''
        Same as get_1gram_dict, but with bigrams (i.e. (w, w') pairs) 
        instead of unigrams.

        all_node_word_dict = {
            node: {
                bigram: (total probability mass of paths from sos to node that contain bigram,
                         number of paths from sos to node that contain bigram)
            }
        }
        '''
        path_count_dict = self._get_node_path_count_dict(all_node_length_dict)

        all_node_word_dict = {node: {} for node in self.nodes}

        visited = {self.sos}
        def dfs_helper(node):
            if node in visited:
                return all_node_word_dict[node]
            visited.add(node)

            curr_word = self.nodes[node]['text']

            curr_word_dict = {} # word -> (total score, count)
            for parent_node, edge_lprob in self.reverse_edges[node].items():
                parent_word_dict = dfs_helper(parent_node)
                
                parent_word = self.nodes[parent_node]['text']
                curr_bigram = (parent_word, curr_word)

                for bigram, (parent_lprob, parent_count) in parent_word_dict.items():
                    if bigram != curr_bigram:
                        added_lprob = parent_lprob + edge_lprob
                        if bigram in curr_word_dict:
                            old_lprob, old_count = curr_word_dict[bigram]
                            curr_word_dict[bigram] = (np.logaddexp(old_lprob, added_lprob),
                                                      old_count + parent_count)
                        else:
                            curr_word_dict[bigram] = (added_lprob, parent_count)
                
                old_lprob, old_count = curr_word_dict.get(curr_bigram, (-float('inf'), 0))
                if target_length is not None:
                    parent_lprob, parent_count = self._get_path_count_bounded_length(all_node_length_dict[parent_node], target_length - (length_to_here + 1), allowed_deviation)
                else:
                    parent_lprob, parent_count = path_count_dict[parent_node] 
                added_lprob = parent_lprob + edge_lprob
                curr_word_dict[curr_bigram] = (np.logaddexp(old_lprob, added_lprob),
                                               old_count + parent_count)

            all_node_word_dict[node] = curr_word_dict
            return curr_word_dict

        for eos in self.eos_list:
            dfs_helper(eos)
        
        word_dict = self._extract_word_dict(all_node_word_dict)
        return word_dict, all_node_word_dict

    def get_2gram_dict_count_aware(self, all_node_length_dict, target_length=None, allowed_deviation=0):
        '''
        Same as get_1gram_dict_count_aware but with bigrams instead of unigrams.

        all_node_word_dict = {
            node: {
                bigram: (total probability mass of paths from sos to node that contain bigram,
                         number of paths from sos to node that contain bigram)
            }
        }
        '''
        raise NotImplementedError
        path_count_dict = self._get_node_path_count_dict(all_node_length_dict)

        all_node_word_dict = {node: {} for node in self.nodes}

        visited = {self.sos}
        def dfs_helper(node):
            if node in visited:
                return all_node_word_dict[node]
            visited.add(node)

            curr_word = self.nodes[node]['text']

            curr_word_dict = {} # word -> (total score, count)
            for parent_node, edge_lprob in self.reverse_edges[node].items():
                parent_word_dict = dfs_helper(parent_node)
                
                parent_word = self.nodes[parent_node]['text']
                curr_bigram = (parent_word, curr_word)

                for bigram, (parent_lprob, parent_count) in parent_word_dict.items():
                    added_lprob = parent_lprob + edge_lprob
                    if bigram in curr_word_dict:
                        old_lprob, old_count = curr_word_dict[bigram]
                        curr_word_dict[bigram] = (np.logaddexp(old_lprob, added_lprob),
                                                    old_count + parent_count)
                    else:
                        curr_word_dict[bigram] = (added_lprob, parent_count)
                
                old_lprob, old_count = curr_word_dict.get(curr_bigram, (-float('inf'), 0))
                parent_lprob, parent_count = path_count_dict[parent_node] 
                added_lprob = parent_lprob + edge_lprob
                curr_word_dict[curr_bigram] = (np.logaddexp(old_lprob, added_lprob),
                                               old_count + parent_count)

            all_node_word_dict[node] = curr_word_dict
            return curr_word_dict

        for eos in self.eos_list:
            dfs_helper(eos)
        
        word_dict = self._extract_word_dict(all_node_word_dict)
        return word_dict, all_node_word_dict

    def get_ngram_dict(self, all_node_length_dict):
        '''
        Same as get_1gram_dict, but with bigrams (i.e. (w, w') pairs) 
        instead of unigrams.

        all_node_word_dict = {
            node: {
                bigram: (total probability mass of paths from sos to node that contain bigram,
                         number of paths from sos to node that contain bigram)
            }
        }
        '''
        path_count_dict = self._get_node_path_count_dict(all_node_length_dict)

        all_node_word_dict = {node: {} for node in self.nodes}

        visited = {self.sos}
        def dfs_helper(node):
            if node in visited:
                return all_node_word_dict[node]
            visited.add(node)

            curr_word = self.nodes[node]['text']

            curr_word_dict = {} # word -> (total score, count)

            parent_bigrams = {
                (self.nodes[parent_node]['text'], curr_word) 
                for parent_node in self.reverse_edges[node]
            }
            for parent_node, edge_lprob in self.reverse_edges[node].items():
                parent_word_dict = dfs_helper(parent_node)
                
                parent_word = self.nodes[parent_node]['text']
                curr_bigram = (parent_word, curr_word)

                for bigram, (parent_lprob, parent_count) in parent_word_dict.items():
                    if bigram not in parent_bigrams:
                        added_lprob = parent_lprob + edge_lprob
                        if bigram in curr_word_dict:
                            old_lprob, old_count = curr_word_dict[bigram]
                            curr_word_dict[bigram] = (np.logaddexp(old_lprob, added_lprob),
                                                      old_count + parent_count)
                        else:
                            curr_word_dict[bigram] = (added_lprob, parent_count)
                
                old_lprob, old_count = curr_word_dict.get(curr_bigram, (-float('inf'), 0))
                parent_lprob, parent_count = path_count_dict[parent_node] 
                added_lprob = parent_lprob + edge_lprob
                curr_word_dict[curr_bigram] = (np.logaddexp(old_lprob, added_lprob),
                                               old_count + parent_count)

            all_node_word_dict[node] = curr_word_dict
            return curr_word_dict

        for eos in self.eos_list:
            dfs_helper(eos)
        
        word_dict = self._extract_word_dict(all_node_word_dict)
        return word_dict, all_node_word_dict

    def _extract_top_gain_path(
        self, 
        all_node_rouge_dict, 
        min_length=0, 
        max_length=float('inf')
    ) -> Tuple[List[str], float]:
        top_gain, top_eos_len = -float('inf'), None
        for eos in self.eos_list:
            for length, (_, gain, _, _) in all_node_rouge_dict[eos].items():
                if not (min_length <= length <= max_length):
                    continue
                if gain > top_gain:
                    top_gain = gain
                    top_eos_len = (eos, length)
        assert top_eos_len is not None
        curr_node, curr_length = top_eos_len
        rev_path = [curr_node]
        while curr_node != self.sos:
            curr_node = all_node_rouge_dict[curr_node][curr_length][-1]
            curr_length -= 1
            rev_path.append(curr_node)
        return rev_path[::-1], top_gain

    def _extract_topk_gain_paths(
        self, 
        all_node_topk_rouge_dict, 
        min_length=0, 
        max_length=float('inf'),
        topk=1
    ):
        topk_eos = heapdict()
        for eos in self.eos_list:
            for length, topk_dict in all_node_topk_rouge_dict[eos].items():
                if not (min_length <= length <= max_length):
                    continue
                for idx, entry in topk_dict.items():
                    topk_eos[(eos, idx, length)] = entry[0]
                    if len(topk_eos) > topk:
                        topk_eos.popitem()
        assert 0 < len(topk_eos) <= topk

        topk_paths = []
        for topk_entry in topk_eos.keys():
            path = []
            curr_node, curr_idx, curr_length = topk_entry
            while curr_node != self.sos:
                path.append(curr_node)
                (*_, curr_node, curr_idx) = all_node_topk_rouge_dict[curr_node][curr_length][curr_idx]
                curr_length -= 1
            path.reverse()
            topk_paths.append(path)
        return topk_paths, list(topk_eos.values())

    def _extract_random_gain_paths(
        self, 
        all_node_topk_rouge_dict, 
        min_length=0, 
        max_length=float('inf'),
        k=1
    ):
        topk_eos = heapdict()
        for eos in self.eos_list:
            for length, topk_dict in all_node_topk_rouge_dict[eos].items():
                if not (min_length <= length <= max_length):
                    continue
                for idx, entry in topk_dict.items():
                    topk_eos[(eos, idx, length)] = random.random()
                    if len(topk_eos) > k:
                        topk_eos.popitem()
        assert 0 < len(topk_eos) <= k

        topk_paths = []
        for topk_entry in topk_eos.keys():
            path = []
            curr_node, curr_idx, curr_length = topk_entry
            while curr_node != self.sos:
                path.append(curr_node)
                (*_, curr_node, curr_idx) = all_node_topk_rouge_dict[curr_node][curr_length][curr_idx]
                curr_length -= 1
            path.reverse()
            topk_paths.append(path)
        return topk_paths, list(topk_eos.values())

    def get_top_rouge1_path(
        self, 
        mean_length: float, 
        exp_word_match: Dict[str, float], 
        d_length=float('inf'), 
        uniform=False,
        lattice_topk=1,
        return_topk=-1,
        use_rouge=True
    ):
        '''
        all_node_rouge_dict = {
            node: {
                length: (max expected rouge over all paths from sos to node, 
                         max weighted sum of gains over all paths from sos to node,
                         E[m(c, h)],
                         parent in max rouge path,
                         parent path idx)
            }
        }

        With slight modification to accomodate top-k instead of single top path:
        length: heapdict({
            curr_idx: (E[sum of gains], E[rouge], E[m(c,h)], parent, parent_idx)
        })
        This uses the fact that tuple comparison in Python is lexicographic 
        (i.e. first items are compared, then second, etc) so this ensures
        that our minheap still operates with expected gain as the key (with
        ties broken arbitrarily).
        '''
        all_node_rouge_dict = {node: {} for node in self.nodes}
        all_node_rouge_dict[self.sos] = {0: heapdict({0: (0, 0, 0, None, 0)})}
        
        visited = {self.sos}
        def dfs_helper(node):
            if node in visited:
                return all_node_rouge_dict[node]
            visited.add(node)
            curr_word = self.nodes[node]['text']
            curr_exp_match = exp_word_match.get(curr_word, 0) #TODO: this feels risky

            curr_rouge_dict_indices = Counter() # length -> idx
            curr_rouge_dict = {}
            for parent, logprob in self.reverse_edges[node].items():
                parent_rouge_dict = dfs_helper(parent)
                for parent_length, parent_topk_paths in parent_rouge_dict.items():
                    for parent_idx, parent_entry in parent_topk_paths.items():
                        (parent_score, parent_rouge, parent_match, *_) = parent_entry

                        new_length = parent_length + 1
                        new_match = parent_match + curr_exp_match
                        new_rouge = 2 * new_match / (new_length + mean_length)

                        if use_rouge:
                            gain = new_rouge - parent_rouge
                        else:
                            gain = curr_exp_match
                        if uniform:
                            logprob = 0
                        new_score = parent_score + gain * np.exp(logprob)
                        
                        if new_length not in curr_rouge_dict:
                            curr_rouge_dict[new_length] = heapdict({
                                0: (new_score, new_rouge, new_match, parent, parent_idx)
                            })
                            curr_rouge_dict_indices[new_length] += 1
                        else:
                            curr_len_topk = curr_rouge_dict[new_length]
                            curr_idx = curr_rouge_dict_indices[new_length]
                            curr_rouge_dict_indices[new_length] += 1
                            curr_len_topk[curr_idx] = (new_score, new_rouge, new_match, parent, parent_idx)
                            if len(curr_len_topk) > lattice_topk:
                                curr_len_topk.popitem()

            all_node_rouge_dict[node] = curr_rouge_dict
            return curr_rouge_dict
        
        for eos in self.eos_list:
            dfs_helper(eos)

        if return_topk < 1:
            return_topk = lattice_topk
        topk_paths, topk_rouges = self._extract_topk_gain_paths(
            all_node_rouge_dict,
            min_length=mean_length-d_length, 
            max_length=mean_length+d_length, 
            topk=return_topk
        )
        return topk_paths, topk_rouges, all_node_rouge_dict

    def get_top_rouge1_path_count_aware(
        self, 
        mean_length: float, 
        exp_word_match: Dict[str, float], 
        d_length=float('inf'), 
        uniform=False,
        lattice_topk=1,
        return_topk=-1, 
        use_rouge=True
    ):
        '''
        Computes path that maximizes sum of local gains

        Note that this is a **greedy approximation** of the optimal path. Under
        count-aware, previous words affect the gain of the next word (i.e. 
        if the next word appeared previously in the sequence, it will have
        lower probability). Hence, picking locally optimal next words will not
        guarantee the best sequence overall and the optimal substructure 
        property may not apply. Nevertheless, this approximation seems to work
        decently well in practice and is much more efficient than an exact
        brute force search over all sequences.

        all_node_rouge_dict = {
            node: {
                length: (max expected rouge over all paths from sos to node, 
                         max weighted sum of gains over all paths from sos to node,
                         E[m(c, h)],
                         parent in max rouge path)
            }
        }

        With slight modification to accomodate top-k instead of single top path:
        length: heapdict({
            (curr_idx): (E[sum of gains], E[rouge], E[m(c,h)], parent, parent_idx)
        })
        '''
        all_node_rouge_dict = {node: {} for node in self.nodes}
        all_node_rouge_dict[self.sos] = {0: heapdict({0: (0, 0, 0, None, 0)})}

        visited = {self.sos}
        def dfs_helper(node):
            if node in visited:
                return all_node_rouge_dict[node]
            visited.add(node)
            curr_word = self.nodes[node]['text']

            curr_rouge_dict_indices = Counter() # length -> idx
            curr_rouge_dict = {}
            for parent, logprob in self.reverse_edges[node].items():
                parent_rouge_dict = dfs_helper(parent)
                for parent_length, parent_topk_paths in parent_rouge_dict.items():
                    for parent_idx, parent_entry in parent_topk_paths.items():
                        (parent_score, parent_rouge, parent_match, *_) = parent_entry
                        
                        new_length = parent_length + 1
                        curr_word_prefix_count = 1 + (self.nodes[parent]['text'] == curr_word)
                        curr_node, curr_idx, curr_length = parent, parent_idx, parent_length
                        while curr_node != self.sos:
                            (*_, curr_node, curr_idx) = all_node_rouge_dict[curr_node][curr_length][curr_idx]
                            curr_length -= 1
                            curr_word_prefix_count += (self.nodes[curr_node]['text'] == curr_word)

                        curr_exp_match = exp_word_match.get((curr_word, curr_word_prefix_count), 0.0)

                        new_match = parent_match + curr_exp_match
                        new_rouge = 2 * new_match / (new_length + mean_length)

                        gain = new_rouge - parent_rouge
                        if uniform:
                            logprob = 0
                        new_score = parent_score + gain * np.exp(logprob)
                        
                        if new_length not in curr_rouge_dict:
                            assert curr_rouge_dict_indices[new_length] == 0
                            curr_rouge_dict[new_length] = heapdict({
                                0: (new_score, new_rouge, new_match, parent, parent_idx)
                            })
                            curr_rouge_dict_indices[new_length] += 1
                        else:
                            curr_len_topk = curr_rouge_dict[new_length]
                            curr_idx = curr_rouge_dict_indices[new_length]
                            curr_rouge_dict_indices[new_length] += 1
                            curr_len_topk[curr_idx] = (new_score, new_rouge, new_match, parent, parent_idx)
                            if len(curr_len_topk) > lattice_topk:
                                curr_len_topk.popitem()

            all_node_rouge_dict[node] = curr_rouge_dict
            return curr_rouge_dict
        
        for eos in self.eos_list:
            dfs_helper(eos)

        if return_topk < 1:
            return_topk = lattice_topk
        topk_paths, topk_rouges = self._extract_topk_gain_paths(
            all_node_rouge_dict,
            min_length=mean_length - d_length, 
            max_length=mean_length + d_length,
            topk=return_topk
        )
        return topk_paths, topk_rouges, all_node_rouge_dict

    def get_top_rouge2_path(
        self, 
        mean_length: float, 
        exp_word_match: Dict[str, float], 
        d_length=float('inf'), 
        uniform=False,
        lattice_topk=1,
        return_topk=-1,
        use_rouge=True
    ):
        '''
        all_node_rouge_dict = {
            node: {
                length: (max expected rouge over all paths from sos to node, 
                         max weighted sum of gains over all paths from sos to node,
                         E[m(c, h)],
                         parent in max rouge path)
            }
        }
        '''
        all_node_rouge_dict = {node: {} for node in self.nodes}
        all_node_rouge_dict[self.sos] = {0: heapdict({0: (0, 0, 0, None, 0)})}
        
        visited = {self.sos}
        def dfs_helper(node):
            if node in visited:
                return all_node_rouge_dict[node]
            visited.add(node)
            curr_word = self.nodes[node]['text']

            curr_rouge_dict_indices = Counter() # length -> idx
            curr_rouge_dict = {}
            for parent, logprob in self.reverse_edges[node].items():
                parent_word = self.nodes[parent]['text']
                curr_exp_match = exp_word_match[(parent_word, curr_word)]
                parent_rouge_dict = dfs_helper(parent)
                for parent_length, parent_topk_paths in parent_rouge_dict.items():
                    for parent_idx, parent_entry in parent_topk_paths.items():
                        (parent_score, parent_rouge, parent_match, *_) = parent_entry

                        new_length = parent_length + 1
                        new_match = parent_match + curr_exp_match
                        new_rouge = 2 * new_match / (new_length + mean_length)

                        if use_rouge:
                            gain = new_rouge - parent_rouge
                        else:
                            gain = curr_exp_match
                        if uniform:
                            logprob = 0
                        new_score = parent_score + gain * np.exp(logprob)

                        if new_length not in curr_rouge_dict:
                            assert curr_rouge_dict_indices[new_length] == 0
                            curr_rouge_dict[new_length] = heapdict({
                                0: (new_score, new_rouge, new_match, parent, parent_idx)
                            })
                            curr_rouge_dict_indices[new_length] += 1
                        else:
                            curr_len_topk = curr_rouge_dict[new_length]
                            curr_idx = curr_rouge_dict_indices[new_length]
                            curr_rouge_dict_indices[new_length] += 1
                            curr_len_topk[curr_idx] = (new_score, new_rouge, new_match, parent, parent_idx)
                            if len(curr_len_topk) > lattice_topk:
                                curr_len_topk.popitem()

            all_node_rouge_dict[node] = curr_rouge_dict
            return curr_rouge_dict
        
        for eos in self.eos_list:
            dfs_helper(eos)

        if return_topk < 1:
            return_topk = lattice_topk
        best_path, best_rouge = self._extract_topk_gain_paths(
            all_node_rouge_dict,
            min_length=mean_length-d_length, 
            max_length=mean_length+d_length, 
            topk=return_topk
        )
        return best_path, best_rouge, all_node_rouge_dict

    def get_top_rouge2_path_count_aware(
        self, 
        mean_length: float, 
        exp_word_match: Dict[str, float], 
        d_length=float('inf'), 
        uniform=False
    ):
        raise NotImplementedError

    def get_top_rougeN_path(
        self, 
        mean_length: float, 
        exp_word_match: Dict[str, float], 
        d_length=float('inf'), 
        uniform=False,
        N=3,
        lattice_topk=1,
        return_topk=-1
    ):
        '''
        all_node_rouge_dict = {
            node: {
                length: (max expected rouge over all paths from sos to node, 
                         max weighted sum of gains over all paths from sos to node,
                         E[m(c, h)],
                         parent in max rouge path)
            }
        }
        '''
        all_node_rouge_dict = {node: {} for node in self.nodes}
        all_node_rouge_dict[self.sos] = {0: heapdict({0: (0, 0, 0, None, 0)})}
        
        visited = {self.sos}
        def dfs_helper(node):
            if node in visited:
                return all_node_rouge_dict[node]
            visited.add(node)
            curr_word = self.nodes[node]['text']

            curr_rouge_dict_indices = Counter() # length -> idx
            curr_rouge_dict = {}
            for parent, logprob in self.reverse_edges[node].items():
                parent_word = self.nodes[parent]['text']
                parent_rouge_dict = dfs_helper(parent)
                for parent_length, parent_topk_paths in parent_rouge_dict.items():
                    for parent_idx, parent_entry in parent_topk_paths.items():
                        (parent_score, parent_rouge, parent_match, *_) = parent_entry

                        ngram = [curr_word, parent_word]

                        traversal_node, traversal_idx = parent, parent_idx
                        traversal_length = parent_length
                        while len(ngram) < N and traversal_node != self.sos:
                            traversal_node, traversal_idx = \
                                all_node_rouge_dict[traversal_node][traversal_length][parent_idx][-2:]
                            ngram.append(self.nodes[traversal_node]['text'])
                        while len(ngram) < N:
                            ngram.append(None)
                        ngram.reverse()
                        ngram = tuple(ngram)
                        curr_exp_match = exp_word_match[ngram]

                        new_length = parent_length + 1
                        new_match = parent_match + curr_exp_match
                        new_rouge = 2 * new_match / (new_length + mean_length)

                        gain = new_rouge - parent_rouge
                        if uniform:
                            logprob = 0
                        new_score = parent_score + gain * np.exp(logprob)

                        if new_length not in curr_rouge_dict:
                            assert curr_rouge_dict_indices[new_length] == 0
                            curr_rouge_dict[new_length] = heapdict({
                                0: (new_score, new_rouge, new_match, parent, parent_idx)
                            })
                            curr_rouge_dict_indices[new_length] += 1
                        else:
                            curr_len_topk = curr_rouge_dict[new_length]
                            curr_idx = curr_rouge_dict_indices[new_length]
                            curr_rouge_dict_indices[new_length] += 1
                            curr_len_topk[curr_idx] = (new_score, new_rouge, new_match, parent, parent_idx)
                            if len(curr_len_topk) > lattice_topk:
                                curr_len_topk.popitem()

            all_node_rouge_dict[node] = curr_rouge_dict
            return curr_rouge_dict
        
        for eos in self.eos_list:
            dfs_helper(eos)

        if return_topk < 1:
            return_topk = lattice_topk
        best_path, best_rouge = self._extract_topk_gain_paths(
            all_node_rouge_dict,
            min_length=mean_length-d_length, 
            max_length=mean_length+d_length, 
            topk=return_topk
        )
        return best_path, best_rouge, all_node_rouge_dict

    def get_top_log_rouge1_path(
        self, 
        mean_length: float, 
        exp_word_match: Dict[str, float], 
        d_length=float('inf'), 
        uniform=False,
        lattice_topk=1,
        return_topk=-1
    ):
        '''
        all_node_rouge_dict = {
            node: {
                length: (max expected rouge over all paths from sos to node, 
                         max weighted sum of gains over all paths from sos to node,
                         E[m(c, h)],
                         parent in max rouge path,
                         parent path idx)
            }
        }

        With slight modification to accomodate top-k instead of single top path:
        length: heapdict({
            curr_idx: (E[sum of gains], E[rouge], E[m(c,h)], parent, parent_idx)
        })
        This uses the fact that tuple comparison in Python is lexicographic 
        (i.e. first items are compared, then second, etc) so this ensures
        that our minheap still operates with expected gain as the key (with
        ties broken arbitrarily).
        '''
        all_node_rouge_dict = {node: {} for node in self.nodes}
        all_node_rouge_dict[self.sos] = {0: heapdict({0: (0, 0, 0, None, 0)})}
        
        visited = {self.sos}
        def dfs_helper(node):
            if node in visited:
                return all_node_rouge_dict[node]
            visited.add(node)
            curr_word = self.nodes[node]['text']
            curr_exp_match = exp_word_match[curr_word]

            curr_rouge_dict_indices = Counter() # length -> idx
            curr_rouge_dict = {}
            for parent, logprob in self.reverse_edges[node].items():
                parent_rouge_dict = dfs_helper(parent)
                for parent_length, parent_topk_paths in parent_rouge_dict.items():
                    for parent_idx, parent_entry in parent_topk_paths.items():
                        (parent_score, parent_log_rouge, parent_match, *_) = parent_entry

                        new_length = parent_length + 1
                        new_match = parent_match + curr_exp_match
                        # new_rouge = 2 * new_match / (new_length + mean_length)
                        new_log_rouge = np.log(new_match) - np.log(new_length + mean_length)

                        gain = new_log_rouge - parent_log_rouge
                        if uniform:
                            logprob = 0
                        new_score = parent_score + gain * np.exp(logprob)
                        
                        if new_length not in curr_rouge_dict:
                            curr_rouge_dict[new_length] = heapdict({
                                0: (new_score, new_log_rouge, new_match, parent, parent_idx)
                            })
                            curr_rouge_dict_indices[new_length] += 1
                        else:
                            curr_len_topk = curr_rouge_dict[new_length]
                            curr_idx = curr_rouge_dict_indices[new_length]
                            curr_rouge_dict_indices[new_length] += 1
                            curr_len_topk[curr_idx] = (new_score, new_log_rouge, new_match, parent, parent_idx)
                            if len(curr_len_topk) > lattice_topk:
                                curr_len_topk.popitem()

            all_node_rouge_dict[node] = curr_rouge_dict
            return curr_rouge_dict
        
        for eos in self.eos_list:
            dfs_helper(eos)

        if return_topk < 1:
            return_topk = lattice_topk
        topk_paths, topk_rouges = self._extract_topk_gain_paths(
            all_node_rouge_dict,
            min_length=mean_length-d_length, 
            max_length=mean_length+d_length, 
            topk=return_topk
        )
        return topk_paths, topk_rouges, all_node_rouge_dict

    def get_path_tokens(self, path):
        return [self.nodes[node]['tok_idx'] for node in path]

    def min_edge_path(self, start, end):
        '''
        Finds path from start node to end node using fewest number of edges
        (using breadth-first search)
        Used only for small ablation on path length (checking if arbitrarily 
        decreasing path length leads to gains in performance; I found that it 
        does not)
        '''
        if start == end:
            return [start]
        visited = {start: None} # tracks visited nodes along with their parent
        frontier = {start}

        def extract_path():
            rev_path = [end]
            curr_node = end
            while curr_node != start:
                curr_node = visited[curr_node]
                rev_path.append(curr_node)
            return rev_path[::-1]

        for _ in range(len(self.nodes)):
            new_frontier = set()
            for node in frontier:
                for child in self.edges[node]:
                    if child not in visited:
                        new_frontier.add(child)
                        visited[child] = node
                    if child == end:
                        return extract_path()
            frontier = new_frontier
        return None # no path found
