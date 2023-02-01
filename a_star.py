import numpy as np
from matplotlib import pyplot as plt

class a_star:
    def __init__(self, num_pairs, rotate_maze=False):
        self.num_pairs = num_pairs
        self.rotate_maze = rotate_maze
        self.best_avg_steps = np.inf
        self.best_avg_steps_not_inf = False
        self.costs_array_set = set()
        self.rng = np.random.default_rng()
        
    def generate_maze(self, maze_size, cutoff, show_graphs=False):
        import cv2
        (self.msy, self.msx) = maze_size
        
        maze = self.rng.random((self.msy,self.msx))
        
        maze[maze>cutoff] = 1
        maze[maze<=cutoff] = 0
        
        kernel = np.ones((3,3), dtype=np.uint8)
        operation = cv2.MORPH_CLOSE
        self.maze = cv2.morphologyEx(maze, operation, kernel)
        
        if show_graphs:
            plt.imshow(self.maze)
            plt.show()
            plt.imshow(np.rot90(self.maze))
            plt.show()
    
    def load_map_from_file(self, map_file_location, show_graphs=False):
        with open(map_file_location, 'r') as f:
            file_str = f.read()
        
        replace_list = [('T', '1 '), ('@', '1 '), ('.', '0 '), ('\n', '')]
        
        for replace_tuple in replace_list:
            (old_str, new_str) = replace_tuple
            file_str = file_str.replace(old_str, new_str)
        
        header = file_str[:file_str.find('map')+3]
        self.msy = int(header[header.find('height')+7:header.find('width')])
        self.msx = int(header[header.find('width')+6:header.find('map')])
        
        file_str = file_str[file_str.find('map')+3:]
        
        map_array_flat = np.fromstring(file_str, sep=' ')
        
        self.maze = map_array_flat.reshape((self.msy, self.msx))
        
        if show_graphs:
            plt.imshow(self.maze)
            plt.show()
        
    def generate_start_goal_lists(self):
        start_idx = self.rng.choice(len(np.where(self.maze==0)[0]), size=self.num_pairs)
        
        starts_y = np.where(self.maze==0)[0][start_idx]
        starts_x = np.where(self.maze==0)[1][start_idx]
        
        starts_list = []
        for i in range(len(starts_y)):
            starts_list.append((starts_y[i], starts_x[i]))
        
        end_idx = self.rng.choice(len(np.where(self.maze==0)[0]), size=self.num_pairs)
        
        ends_y = np.where(self.maze==0)[0][end_idx]
        ends_x = np.where(self.maze==0)[1][end_idx]
        
        ends_list = []
        for i in range(len(ends_y)):
            ends_list.append((ends_y[i], ends_x[i]))
        
        self.starts = starts_list
        self.ends = ends_list
        
        print(self.starts)
        print(self.ends)
    
    def additional_gets(self):
        # state matrices
        self.state_x = np.vstack((np.arange(self.msx),) * self.msy)
        self.state_y = np.hstack((np.arange(self.msy).reshape(-1,1),) * self.msx)
        
        # rotated (start,goal) pairs
        if self.rotate_maze:
            self.rotated_starts = []
            self.rotated_ends = []
            for i in range(len(self.starts)):
                temp_mat = np.zeros((self.msy, self.msx))
                temp_mat[self.starts[i]] = 1
                temp_mat[self.ends[i]] = -1
                temp_mat = np.rot90(temp_mat, k=1)
                start_idx = int(np.argmax(temp_mat))
                end_idx = int(np.argmin(temp_mat))
                true_start = (start_idx // self.msy, start_idx % self.msy)
                true_end = (end_idx // self.msy, end_idx % self.msy)
                self.rotated_starts.append(true_start)
                self.rotated_ends.append(true_end)
        
        # cutouts for duplicate checking
        self.part_idx_dict = {}
        for i in range(len(self.starts)):
            for j,x in enumerate(['row', 'col']):
                if x == 'row':
                    maze_dim_size = self.msy
                elif x == 'col':
                    maze_dim_size = self.msx
                if self.starts[i][j] < 2:
                    self.part_idx_dict[f'{i}_{x}_start'] = self.starts[i][j]
                    self.part_idx_dict[f'{i}_{x}_end'] = self.starts[i][j] + 5
                elif self.starts[i][j] > maze_dim_size - 4:
                    self.part_idx_dict[f'{i}_{x}_start'] = maze_dim_size - 6
                    self.part_idx_dict[f'{i}_{x}_end'] = maze_dim_size - 1
                else:
                    self.part_idx_dict[f'{i}_{x}_start'] = self.starts[i][j] - 2
                    self.part_idx_dict[f'{i}_{x}_end'] = self.starts[i][j] + 3

    def run_a_star(self, heuristic, compare_normalized_costs=False, update_best=True, show_graphs=False):
        # have input for heuristic
        # have this function be for one A* run, separate function to call them all
        total_num_steps = 0
        
        # generate 3D array of costs
        for i in range(len(self.starts)):
            env = {
                'state_x': self.state_x,
                'state_y': self.state_y,
                'goal_x':  self.ends[i][1],
                'goal_y':  self.ends[i][0],
            }
            
            costs_array = heuristic.interpret(env)
            costs_array_normalized = costs_array - np.min(costs_array)
            if np.max(costs_array_normalized) != 0:
                costs_array_normalized = costs_array_normalized / np.max(costs_array_normalized)
            try:
                if not costs_array.shape:
                    return np.inf, False, None
            except AttributeError:
                return np.inf, False, None
            
            r1 = self.part_idx_dict[f'{i}_row_start']
            r2 = self.part_idx_dict[f'{i}_row_end']
            c1 = self.part_idx_dict[f'{i}_col_start']
            c2 = self.part_idx_dict[f'{i}_col_end']
            
            if compare_normalized_costs:
                costs_part = costs_array_normalized[r1:r2, c1:c2]
            else:
                costs_part = costs_array[r1:r2, c1:c2]
            
            if i == 0:
                costs_array_full = costs_array_normalized
                costs_part_full = costs_part
            elif i == 1:
                costs_array_full = np.stack((costs_array_full, costs_array_normalized), axis=0)
                costs_part_full = np.stack((costs_part_full, costs_part), axis=0)
            else:
                costs_array_full = np.concatenate((costs_array_full, np.expand_dims(costs_array_normalized, axis=0)), axis=0)
                costs_part_full = np.concatenate((costs_part_full, np.expand_dims(costs_part, axis=0)), axis=0)
                
        # make sure max of costs array doesn't exceed 555,555,555
        # can change stuff to np.inf and use nanmax / nanmin
        if np.max(costs_array_full) > 1:
            print('max cost exceeded')
            return np.inf, False, None
        
        # check if costs array is a copy
        if update_best:
            if tuple(costs_part_full.flatten()) in self.costs_array_set:
                return np.inf, False, None
            else:
                self.costs_array_set.add(tuple(costs_part_full.flatten()))

        # number of (start,goal) pairs
        if self.rotate_maze:
            pair_num = 2 * len(self.starts)
        else:
            pair_num = len(self.starts)
        
        # actually running A*
        for i in range(pair_num):
            # which index in cost array to use
            if self.rotate_maze:
                costs_array_idx = i // 2
            else:
                costs_array_idx = i
            
            costs_array = costs_array_full[costs_array_idx,:,:]
            costs_array[self.maze==1] = 9
            costs_array_padded = np.pad(costs_array, 1, mode='constant', constant_values=9)
            
            # whether the maze is rotated
            if self.rotate_maze:
                if i % 2 == 0:
                    rotated = False
                elif i % 2 == 1:
                    rotated = True
            else:
                rotated = False
                
            # true start and end, accounting for padding and rotation
            if rotated:
                costs_array_padded = np.rot90(costs_array_padded, k=1)
                
                true_start = (self.rotated_starts[costs_array_idx][0]+1, self.rotated_starts[costs_array_idx][1]+1)
                true_end = (self.rotated_ends[costs_array_idx][0]+1, self.rotated_ends[costs_array_idx][1]+1)
                
                true_msy = self.msx + 2
                true_msx = self.msy + 2
                
            else:
                true_start = (self.starts[costs_array_idx][0]+1, self.starts[costs_array_idx][1]+1)
                true_end = (self.ends[costs_array_idx][0]+1, self.ends[costs_array_idx][1]+1)
                
                true_msy = self.msy + 2
                true_msx = self.msx + 2
            
            revealed_array = np.ones(costs_array_padded.shape) * 9
            
            revealed_array[true_start] = 7
            modifiable_costs_array_padded = np.copy(costs_array_padded)
            
            num_steps = 0
            while True:
                num_steps += 1
                min_idx = self.rng.choice(np.where(revealed_array.flatten() == revealed_array.flatten().min())[0])
                row = min_idx // true_msx
                col = min_idx % true_msx
                if true_end == (row,col):
                    break
                modifiable_costs_array_padded[row,col] = 7
                revealed_array[row-1:row+2,col-1:col+2] = modifiable_costs_array_padded[row-1:row+2,col-1:col+2]
                if num_steps > true_msy*true_msx:
                    print(num_steps)
                    return np.inf, None, None
                if self.best_avg_steps_not_inf:
                    if total_num_steps + num_steps > self.best_avg_steps * self.num_pairs:
                        return np.inf, True, costs_array_idx
            
            if show_graphs:
                modifiable_costs_array_padded[true_start] = 5 # blue
                modifiable_costs_array_padded[true_end] = 3 # purple
                
                plt.imshow(modifiable_costs_array_padded)
                plt.show()
            
            total_num_steps += num_steps
        
        avg_num_steps = total_num_steps / self.num_pairs
        if avg_num_steps < self.best_avg_steps and update_best:
            self.best_avg_steps_not_inf = True
            self.best_avg_steps = avg_num_steps
        
        return avg_num_steps, True, self.num_pairs