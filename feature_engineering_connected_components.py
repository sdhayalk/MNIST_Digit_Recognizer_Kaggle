import numpy as np

def convert_to_0_1(features):
    new_features = []
    for arr in features:
        temp = []
        for element in arr:
            if element >= 0.5:
                temp.append(1)
            else:
                temp.append(0)
        new_features.append(temp)

    return np.array(new_features, dtype='int')


def search_neighbours(temp_x, temp_y, visited, queue, features_reshaped):
    if (temp_y-1 >= 0):
        if (temp_x, temp_y-1) not in visited and features_reshaped[temp_x][temp_y-1] == 0:
            visited.add((temp_x, temp_y-1))
            queue.append((temp_x, temp_y-1))

    if (temp_y+1 < features_reshaped.shape[0]):
        if (temp_x, temp_y+1) not in visited and features_reshaped[temp_x][temp_y+1] == 0:
            visited.add((temp_x, temp_y+1))
            queue.append((temp_x, temp_y+1))

    if (temp_x+1 < features_reshaped.shape[0]):
        if (temp_x+1, temp_y) not in visited and features_reshaped[temp_x+1][temp_y] == 0:
            visited.add((temp_x+1, temp_y))
            queue.append((temp_x+1, temp_y))

    if (temp_x-1 >= 0):
        if (temp_x-1, temp_y) not in visited and features_reshaped[temp_x-1][temp_y] == 0:
            visited.add((temp_x-1, temp_y))
            queue.append((temp_x-1, temp_y))

    return visited, queue


def find_next_unvisited_point(features_reshaped, visited):
    flag = False
    found = False
    new_x = 0
    new_y = 0

    for x in range(0, 28):
        for y in range(0, 28):
            if (x,y) not in visited and not found and features_reshaped[x][y] == 0:
                new_x = x
                new_y = y
                flag = True
                found = True
                break

    return new_x, new_y, flag

def find_batch_connected_components(batch_x):
    batch_x_connected_components = []
    for features in batch_x:
        features = convert_to_0_1(features)
        features_reshaped = features.reshape((28,28))
        flag = True
        visited = []
        queue = []
        visited = set()
        visited.add((0,0))      # (0,0) is root
        queue.append((0,0))     # (0,0) is root
        connected = 0
        new_features = []

        while(flag):
            connected += 1
            while(queue):
                temp_x, temp_y = queue.pop(0)
                visited, queue = search_neighbours(temp_x, temp_y, visited, queue, features_reshaped)

            flag = False
            new_x, new_y, flag = find_next_unvisited_point(features_reshaped, visited)
            visited.add((new_x, new_y))
            queue.append((new_x, new_y))
        
        if connected == 0:
            new_features = [1.0, 0.0, 0.0, 0.0]
        elif connected == 1:
            new_features = [0.0, 1.0, 0.0, 0.0]
        elif connected == 2:
            new_features = [0.0, 0.0, 1.0, 0.0]
        else:
            new_features = [0.0, 0.0, 0.0, 1.0]

        batch_x_connected_components.append(new_features)

    return np.array(batch_x_connected_components, dtype='float')
