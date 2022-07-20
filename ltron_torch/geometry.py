import numpy

def angle_surrogate(x, dim_1=-2, dim_2=-1):
    '''
    Given a tensor of 3x3 rotation matrices, this will return a matrix of
    values from 0 to 1 representing the angle expressed by the rotation matrix.
    A value of 0 corresponds to a 180 degree rotation, a value of 1 corresponds
    to a 0 degree rotation.  This value is not a linear transform of the angle,
    but is monotonically consistent with it.
    '''
    default_index = [slice(None) for _ in x.shape]
    trace = 0
    for i in range(3):
        index = default_index[:]
        index[dim_1] = i
        index[dim_2] = i
        trace = trace + x[tuple(index)]
    
    return (trace + 1.) / 4.

def get_bbox_corners(bbox):
    low, high = bbox
    corners = numpy.zeros((3,8))
    corners[0,:] = [low[0], high[0]] * 4
    corners[1,:] = [low[1], low[1], high[1], high[1]] * 2
    corners[2,:] = [low[2]]*4 + [high[2]]*4
    return corners

def bbox_avg_distance(pose_a, pose_b, bbox):
    corners = get_bbox_corners(bbox)
    corners = numpy.concatenate((corners, numpy.ones((1,8))), axis=0)
    corners_a = pose_a @ corners
    corners_a = corners_a[:3] / corners_a[[3]]
    corners_b = pose_b @ corners
    corners_b = corners_b[:3] / corners_b[[3]]
    
    corner_offset = corners_b - corners_a
    corner_distance = numpy.sum((corner_offset ** 2), axis=0)**0.5
    
    return numpy.mean(corner_distance)
