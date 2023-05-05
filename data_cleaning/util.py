import numpy as np

NUM_JOINTS = 33

def FlipPoints(M, point_pairs, frame_start, frame_end):
    '''
    Flips keypoint data for a given set of point pairs for the 
    frames within [frame_start, frame_end)

    M: array of keypoints
    point_pairs: an array of index pairs to swap data for
    frame_start: the starting frame to start flipping the points
    frame_end: the ending frame to finish flipping the points
    '''

    for i in range(frame_start, frame_end):
        for (point_i, point_j) in point_pairs:
            tmp = np.copy(M[i,point_i,0:2])
            M[i,point_i,0:2] = np.copy(M[i,point_j,0:2])
            M[i,point_j,0:2] = tmp

    return M

def Acce(M, idx):
    '''
    Computing overall acceleration of joins in a given frame

    M: array of keypoints
    idx: frame index to compute acceleration for
    '''

    # M contains all motion data, index is the point at which to compute acceleration
    frames_sum = 0
    for i in range(NUM_JOINTS):
        frames_sum += np.linalg.norm(M[idx+1,i,:] - 2*M[idx,i,:] + M[idx-1,i,:])
    return frames_sum

def AcRDP(M, epsilon, accel_threshold):
    '''
    Runs the acceleration-constrained RDP program
    More information provided in the documentation Wiki for this project

    M: array of keypoints
    epsilon: minimum allowance for distance threshold to be considered two different points
    accel_threshold: maximum allowed percentile of acceleration for filtering unnatural spikes
    '''

    L = [0, len(M) - 1]

    accelerations = [0]
    for i in range(1, len(M)-1):
        accelerations.append(Acce(M, i))
    accelerations.append(0)

    alpha = np.percentile(accelerations, accel_threshold)

    recurse(L, M, 0, len(M), epsilon, alpha, accelerations)
    return L

def lerp(t, times, points):
    '''
    Smoothstep lerping between two given points at two different times

    t: current time
    times: pair of times where the two points occur
    points: pair of points to lerp between
    '''

    dt = (t-times[0]) / (times[1]-times[0])
    interp_val = dt * dt * (3 - 2 * dt) # smoothstep function interpolation

    out_vec = []
    out_dim = len(points[0])
    for i in range(out_dim):
        diff = points[1][i] - points[0][i]
        out_vec.append(interp_val*diff + points[0][i])

    return out_vec

def recurse(L, M, lo, hi, epsilon, alpha, accelerations):
    '''
    Main AcRDP recursive loop: finding the peaks and valleys, preventing acceleration
    spikes, and interpolating points in between

    L: output list of indices to keep
    M: array of keypoints
    lo: lower bound of frame index
    hi: upper bound of frame index
    epsilon: minimum allowance for distance threshold to be considered two different points
    alpha: maximum allowed acceleration for filtering unnatural spikes
    accelerations: list of computed accelerations at each frame 
    '''

    searched = set()
    maxD = 0
    idx = -1

    times = [lo, hi-1]

    while (idx == -1) or (maxD <= epsilon or accelerations[idx] > alpha):
        if hi - lo <= len(searched) + 2:
            return
        
        max_dist = -1
        max_idx = -1

        for test_index in range(lo + 1, hi - 1):
            if test_index in searched:
                continue

            pass_point = False
            for i in range(NUM_JOINTS):
                if M[test_index,i,:][0] == -1:
                    pass_point = True
                    break

            if pass_point:
                continue

            # Compute deviation
            distance_sum = 0
            for i in range(NUM_JOINTS):
                points = [M[lo,i,:], M[hi-1,i,:]]
                interp_point = lerp(test_index, times, points)
                distance_sum += np.linalg.norm(M[test_index,i,:] - interp_point)

            if distance_sum > max_dist:
                max_dist = distance_sum
                max_idx = test_index

        idx = max_idx
        maxD = max_dist

        searched.add(idx)

        if maxD <= epsilon:
            return

    L.append(idx)
    recurse(L, M, lo, idx, epsilon, alpha, accelerations)
    recurse(L, M, idx, hi, epsilon, alpha, accelerations)