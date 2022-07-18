import numpy as np

def feedforward_weights_initiliazation(N_out, N_in, conf):

    option = conf['w_feedforward_init']

    if option=='all_equal':

        return conf['max_init']*np.ones((N_out, N_in))

    elif option == 'convolutional':
        return convolutional_initialization(N_out, N_in, conf, gaus=True)

    elif option == 'random-convolutional':
        return convolutional_initialization(N_out, N_in, conf, rand=True)

    elif option == 'uniform-convolutional':
        return convolutional_initialization(N_out, N_in, conf, unif=True)

    elif option == 'all_ones':
        return np.ones((N_out, N_in))
    
    elif option == 'random':
        return conf['max_init']*np.random.rand(N_out, N_in) 

    elif option =='identity':
        return conf['max_init']*np.eye(N_out)
    
    else:
        print("Invalid option for weights initialization!")
        exit()


def lateral_weights_initialization(N_out, conf):

    option = conf['w_lateral_init']

    if option=='None':
        
        return None
    
    elif option=='standard':

        thetas = 2*np.pi*np.arange(1,N_out+1)*(1./N_out)
        diff_theta = thetas - thetas.reshape(-1,1)
        f = np.exp(conf['psi']*np.cos(diff_theta)) #lateral connectivity function
        np.fill_diagonal(f,0)
        W = conf['w_minus']*(1./N_out) + conf['w_plus']*f/f.sum(axis=0) #lateral connectivity action neurons

        return W

    else:
        print("Invalid option for weights initialization!")
        exit()

def convolutional_initialization(N_out, N_in, conf, gaus=False, rand=False, unif=False):

    maximum = conf['max_init']
    sigma = conf['sigma_init']

    xs = conf['bounds_x']
    ys = conf['bounds_y']
    space = conf['space_pc']
    if conf['offset']:
        x_pc = np.round(np.arange(xs[0], xs[1], space)+space/2,2)
        y_pc = np.round(np.arange(ys[0], ys[1], space)+space/2,2)
    else:
        x_pc = np.round(np.arange(xs[0], xs[1]+space, space),2)
        y_pc = np.round(np.arange(ys[0], ys[1]+space, space),2)
        
    xx, yy = np.meshgrid(x_pc, y_pc)
    pc = np.stack([xx,yy], axis=2).reshape(-1,2)

    weights = np.zeros((N_out, N_in))

    assert N_out == N_in
    

    for i in range(N_out):

        for j in range(N_in):

            dst = np.sqrt(((pc[i] - pc[j])**2).sum())

            if gaus==True:

                weights[i,j] =  maximum*np.exp(-( (dst)**2 / ( 2.0 * sigma**2 ) ) )

            elif dst<sigma:

                    if rand==True:
                        weights[i,j] = maximum*np.random.rand()
                    if unif==True:
                        weights[i,j] = maximum
            


    return weights