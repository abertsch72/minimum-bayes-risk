import sys
sys.path.append("./")
sys.path.append("./src/")

from src.recom_search.evaluation.analysis import find_start_end

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
        self.sos, self.eos_list, _ = find_start_end(node_dict, edge_dict)
        for eos in self.eos_list: # validate that eos nodes don't have any outgoing edges
            assert len(self.edges[eos]) == 0

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
            for length, (length_score, length_count) in all_node_length_dict[eos].items():
                if length in length_dict:
                    old_score, old_count = length_dict[length]
                    length_dict[length] = (old_score + length_score, old_count + length_count)
                else:
                    length_dict[length] = (length_score, length_count)
        return length_dict

    # @lru_cache(maxsize=1)
    def get_length_dict_bfs(self):
        assert not self.check_cycle()
        node_bag = set()
        node_bag.add(self.sos)
        # node_length_dict = {
        #   node: {
        #       length: (score, count)
        #   }
        # }
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
                print(not_done_parents)
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
        all_node_length_dict = {
            node: {
                length: (score, count)
            }
        }
        '''
        all_node_length_dict = {node: {} for node in self.nodes}
        all_node_length_dict[self.sos] = {0: (0, 1)}

        visited = {self.sos}
        def dfs_helper(node):
            if node in visited:
                return all_node_length_dict[node]
            visited.add(node)

            curr_length_dict = {}
            for parent_node, weight in self.reverse_edges[node].items():
                parent_length_dict = dfs_helper(parent_node)
                for length, (length_score, length_count) in parent_length_dict.items():
                    new_length = length + 1
                    added_score = length_score + weight * length_count
                    if new_length in curr_length_dict:
                        old_score, old_count = curr_length_dict[new_length]
                        new_entry = (old_score + added_score,
                                     old_count + length_count)
                    else:
                        new_entry = (added_score, length_count)
                    curr_length_dict[new_length] = new_entry
                
            all_node_length_dict[node] = curr_length_dict
            return curr_length_dict

        for eos in self.eos_list:
            dfs_helper(eos)
        
        length_dict = self._extract_length_dict(all_node_length_dict)
        return length_dict, all_node_length_dict

    def get_parents(self, node):
        return {other for other in self.nodes if node in self.edges[other]}

    def _get_node_path_count_dict(self, all_node_length_dict):
        '''
        Returns dict mapping each node to 2-tuple containing
        (total score of paths from sos token to that node
         # of paths from sos token to that node)
        '''
        path_count_dict = {}
        for node, length_data in all_node_length_dict.items():
            total_score, total_count = 0, 0
            for s, c in length_data.values():
                total_score += s
                total_count += c
            path_count_dict[node] = (total_score, total_count)
        return path_count_dict

    def _extract_word_dict(self, all_node_word_dict):
        word_dict = {}
        for eos in self.eos_list:
            for word, (word_score, word_count) in all_node_word_dict[eos].items():
                old_score, old_count = word_dict.get(word, (0,0))
                word_dict[word] = (old_score + word_score, old_count + word_count)
        return word_dict

    def get_word_dict(self, all_node_length_dict):
        '''
        all_node_word_dict = {
            node: {
                word: (total score of paths from sos to node that contain word,
                       number of paths from sos to node that contain word)
            }
        }
        '''
        path_count_dict = self._get_node_path_count_dict(all_node_length_dict)

        all_node_word_dict = {node: {} for node in self.nodes}

        visited = set()
        def dfs_helper(node):
            if node in visited:
                return all_node_word_dict[node]
            visited.add(node)

            curr_word = self.nodes[node]['text']

            curr_word_dict = {} # word -> (total score, count)
            for parent_node, weight in self.reverse_edges[node].items():
                parent_word_dict = dfs_helper(parent_node)
                for word, (parent_score, parent_count) in parent_word_dict.items():
                    if word != curr_word:
                        old_score, old_count = curr_word_dict.get(word, (0,0))
                        added_score = parent_score + weight * parent_count
                        curr_word_dict[word] = (old_score + added_score,
                                                old_count + parent_count)
            # The number of paths that contain the current word is equal
            # to the number of paths that reach the current node.
            curr_word_dict[curr_word] = path_count_dict[node]

            if 'end' in curr_word_dict:
                print('end:', curr_word_dict['end'])

            all_node_word_dict[node] = curr_word_dict
            return curr_word_dict

        for eos in self.eos_list:
            dfs_helper(eos)
        
        word_dict = self._extract_word_dict(all_node_word_dict)
        return word_dict, all_node_word_dict
