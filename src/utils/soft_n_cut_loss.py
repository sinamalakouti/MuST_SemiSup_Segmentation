import torch

# n_cut params

sigma_I = 10
sigma_X = 4
radius = 5

def compute_weigths(flatten_image, rows, cols, std_intensity=10, std_position=4, radius=5):

    A = outer_product(flatten_image, torch.ones_like(flatten_image))
    A_T = A.T
    intensity_weight = torch.exp(-1 * torch.square((torch.div((A - A_T), std_intensity))))


    xx, yy = torch.meshgrid(torch.range(1,rows), torch.range(1,cols))
    xx = torch.reshape(xx, (rows * cols,))
    yy = torch.reshape(yy, (rows * cols,))
    A_x = outer_product(xx, torch.ones_like(xx))
    A_y = outer_product(yy, torch.ones_like(yy))

    xi_xj = A_x - A_x.T
    yi_yj = A_y - A_y.T

    sq_distance_matrix = torch.square(xi_xj) + torch.square(yi_yj)
    sq_distance_matrix[sq_distance_matrix >= radius] = 0
    dist_weight = torch.exp(-torch.div(sq_distance_matrix, torch.square(torch.tensor(std_position))))
    dist_weight = dist_weight.type(torch.FloatTensor)
    print(dist_weight.shape)
    print(intensity_weight.shape)
    weight = torch.mul(intensity_weight, dist_weight)

    return weight


def outer_product(v1, v2):
    '''
	Inputs:
	v1 : m*1 tf array
	v2 : m*1 tf array
	Output :
	v1 x v2 : m*m array
	'''

    v1 = torch.reshape(v1, (-1,))
    v2 = torch.reshape(v2, (-1,))
    v1 = torch.unsqueeze((v1), axis=0)
    v2 =torch.unsqueeze((v2), axis=0)
    return torch.matmul(v1.T, (v2))


def numerator(k_class_prob, weights):
    '''
    Inputs :
    k_class_prob : k_class pixelwise probability (rows*cols) tensor
    weights : edge weights n*n tensor
    '''
    k_class_prob = torch.reshape(k_class_prob, (-1,))
    return torch.mul(weights, outer_product(k_class_prob, k_class_prob)).sum()


def denominator(k_class_prob, weights):
    '''
    Inputs:
    k_class_prob : k_class pixelwise probability (rows*cols) tensor
    weights : edge weights	n*n tensor
    '''
    k_class_prob = k_class_prob.type(torch.FloatTensor)
    k_class_prob = torch.reshape(k_class_prob, (-1,))
    return torch.mul(weights, outer_product(k_class_prob, torch.ones(k_class_prob.shape))).sum()


def soft_n_cut_loss(weights, prob, k, rows, cols):
    '''
    Inputs:
    prob : (rows*cols*k) tensor
    k : number of classes (integer)
    flatten_image : 1 dim tf array of the row flattened image ( intensity is the average of the three channels)
    rows : number of the rows in the original image
    cols : number of the cols in the original image
    Output :
    soft_n_cut_loss tensor for a single image
    '''

    soft_n_cut_loss = k
    print("here1")
    # weights = edge_weights(flatten_image, rows, cols)

    for t in range(0,k):
        print("here")
        soft_n_cut_loss = soft_n_cut_loss - (numerator(prob[:, :, t], weights) / denominator(prob[:, :, t], weights))

    return soft_n_cut_loss
