import matplotlib.pyplot as plt
import numpy as np

def make_firing_rates_plot(ax, rates):

    plot = ax.imshow(rates, origin='lower', cmap='Blues', interpolation='none',aspect='auto')

    cb = ax.get_figure().colorbar(plot, ax=ax, format='%.1f')

    return cb


def make_action_weights_plot(ax, weights, w_min, w_max):

  N_out, N_in = weights.shape

  num_images = int(np.sqrt(N_out))
  image_side = int(np.sqrt(N_in))

  subfig = ax.get_figure()
  subplots = subfig.subplots(num_images, num_images)

  for i, sp in enumerate(subplots.reshape(-1)):

      sp.axis('off')
      image = weights[i].reshape(-1, image_side)
      im = sp.imshow(image, cmap = 'bwr', vmin=w_min, vmax=w_max, origin='lower')

  fig = ax.get_figure()
  fig.subplots_adjust(right=0.8)

  cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
  cb = fig.colorbar(im, cax=cbar_ax)

  return cb


def compute_length(trajectory):
    """
    Receives a numpy array of shape (num_steps, 2), representing the trajectory of an agent.
    Returns the length of the path.
    """
    
    displacements = trajectory[1:] - trajectory[:-1]
    displacements_length = np.sqrt((displacements**2).sum(axis=1))
    
    return displacements_length.sum()


def make_weights_plot(ax, weights, w_min, w_max):

  N_out, N_in = weights.shape

  num_images = int(np.sqrt(N_out))
  image_side = int(np.sqrt(N_in))

  subfig = ax.get_figure()
  subplots = subfig.subplots(num_images, num_images)

  for i, sp in enumerate(subplots.reshape(-1)):

      sp.axis('off')
      image = weights[i].reshape(-1, image_side)
      im = sp.imshow(image, cmap = 'Blues', vmin=w_min, vmax=w_max, origin='lower')

  fig = ax.get_figure()
  fig.subplots_adjust(right=0.8)

  cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
  cb = fig.colorbar(im, cax=cbar_ax)

  return cb

 

def initialize_plots(CA1, CA3, ENV):


        fig = plt.figure()
        fig.canvas.manager.full_screen_toggle() 

        subfigs = fig.subfigures(1, 3, wspace=0.01, width_ratios=[0.7,1.2,1.1]) 

        subfigs[0].set_facecolor('lemonchiffon')
        subfigs[1].set_facecolor('lightcyan')

        ax0, ax1 = subfigs[0].subplots(2,1, sharex=True)
        ax0.get_yaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)

        ax2,ax3,ax4 = subfigs[1].subplots(3,1, sharex=True)

        subsubfig = subfigs[2].subfigures(2,1, wspace=0.07)
        subsubfig[0].set_facecolor('gray')
        subsubfig[1].set_facecolor('thistle')

        ax5 = subsubfig[0].subplots(1,1)  
        ax6 = subsubfig[1].subplots(1,1)

        ax5.axis('off')
        ax6.axis('off')

        subsubfig[0].subplots_adjust(wspace=0., hspace=0.05)
        subsubfig[1].subplots_adjust(wspace=0., hspace=0.05)


        ax0.plot(ENV.rew_center[0]+ENV.rew_radius*np.cos(np.linspace(-np.pi,np.pi,100)), ENV.rew_center[1]+ENV.rew_radius*np.sin(np.linspace(-np.pi,np.pi,100)),'b') #plot reward 1
        ax0.plot([ENV.bounds_x[0],ENV.bounds_x[1]], [ENV.bounds_y[1],ENV.bounds_y[1]],c='k', ls='--',lw=0.5)
        ax0.plot([ENV.bounds_x[0],ENV.bounds_x[1]], [ENV.bounds_y[0],ENV.bounds_y[0]],c='k', ls='--',lw=0.5)
        ax0.plot([ENV.bounds_x[0],ENV.bounds_x[0]], [ENV.bounds_y[0],ENV.bounds_y[1]],c='k', ls='--',lw=0.5)
        ax0.plot([ENV.bounds_x[1],ENV.bounds_x[1]], [ENV.bounds_y[0],ENV.bounds_y[1]],c='k', ls='--',lw=0.5)
        ax0.scatter(CA1.place_model.pc[:,0],CA1.place_model.pc[:,1], s=1, label="CA1")
        #ax0.scatter(CA1.place_model.pc[104,0],CA1.place_model.pc[104,1], s=140, facecolors='none', edgecolors='r')
        if CA1.place_model.offset != CA3.neuron_model.offset:
            ax0.scatter(CA3.neuron_model.pc[:,0],CA3.neuron_model.pc[:,1], s=1, label ="CA3")

        if ENV.obstacle:
            ax0.plot([ENV.obs_x[0],ENV.obs_x[1]], [ENV.obs_y[1],ENV.obs_y[1]],c='k', ls='--',lw=0.5)
            ax0.plot([ENV.obs_x[0],ENV.obs_x[1]], [ENV.obs_y[0],ENV.obs_y[0]],c='k', ls='--',lw=0.5)
            ax0.plot([ENV.obs_x[0],ENV.obs_x[0]], [ENV.obs_y[0],ENV.obs_y[1]],c='k', ls='--',lw=0.5)
            ax0.plot([ENV.obs_x[1],ENV.obs_x[1]], [ENV.obs_y[0],ENV.obs_y[1]],c='k', ls='--',lw=0.5)

        if ENV.obstacle_2:
            ax0.plot([ENV.obs_x_2[0], ENV.obs_x_2[1]], [ENV.obs_y_2[1],ENV.obs_y_2[1]],c='k', ls='--',lw=0.5)
            ax0.plot([ENV.obs_x_2[0], ENV.obs_x_2[1]], [ENV.obs_y_2[0],ENV.obs_y_2[0]],c='k', ls='--',lw=0.5)
            ax0.plot([ENV.obs_x_2[0], ENV.obs_x_2[0]], [ENV.obs_y_2[0],ENV.obs_y_2[1]],c='k', ls='--',lw=0.5)
            ax0.plot([ENV.obs_x_2[1], ENV.obs_x_2[1]], [ENV.obs_y_2[0],ENV.obs_y_2[1]],c='k', ls='--',lw=0.5)

        ax1.set_title('Policy')
        ax1.plot(ENV.rew_center[0]+ENV.rew_radius*np.cos(np.linspace(-np.pi,np.pi,100)), ENV.rew_center[1]+ENV.rew_radius*np.sin(np.linspace(-np.pi,np.pi,100)),'b') #plot reward 1
        
        ax1.scatter(CA1.place_model.pc[:,0],CA1.place_model.pc[:,1], s=1, label="CA1")
        if CA1.place_model.offset != CA3.neuron_model.offset:
            ax1.scatter(CA3.neuron_model.pc[:,0],CA3.neuron_model.pc[:,1], s=1, label ="CA3")

        ax2.set_title('CA3 firing rates')
        ax3.set_title('CA1 firing rates')
        ax4.set_title('Action Layer firing rates')

        ax5.set_title('CA3-CA1 weights')
        ax6.set_title('CA1-Action weights')

        plt.pause(0.001)

        return fig


def update_plots(fig, trial, store_pos, starting_position,
                 firing_rate_store_AC, firing_rate_store_CA1,
                 firing_rate_store_CA3, CA3, CA1, AC, ENV):

    ax0, ax1, ax2, ax3, ax4, ax5, ax6 = fig.get_axes()[0:7]

    colorbars = []

   #action_ticks = [4+4*i for i in range(9)]
    action_degree = [str(int(x)) for x in list(AC.thetas/(2*np.pi)*360)]


    ###### TRAJECTORY ########

    trajectory = store_pos[trial]
    trajectory = trajectory[(trajectory[:,0]!=0.)|(trajectory[:,1]!=0.)]

    L = compute_length(trajectory)

    ax0.set_title('Trial {} (L={:.1f})'.format(trial, L))
    
    if starting_position is not None:
        ax0.plot(starting_position[0],starting_position[1], marker='$\U0001F42D$', color='black', markersize=20) 

    F1 = ax0.plot(trajectory[:, 0], trajectory[:,1])
    
    ####### POLICY #########

    if CA1.alpha == 0:
        ac = np.dot(AC.actions, AC.neuron_model.W) #vector of preferred actions according to the weights
    elif CA1.alpha == 1:
        ac = np.dot(AC.actions, np.dot(AC.neuron_model.W, CA1.SRM0_model.W))
    else:
        ac = (1-CA1.alpha)*np.dot(AC.actions, AC.neuron_model.W)/AC.a + CA1.alpha*np.dot(AC.actions, np.dot(AC.w_ca1, CA1.SRM0_model.W))


    if CA1.alpha==0:
        f4 = ax1.quiver(CA1.place_model.pc[:,0], CA1.place_model.pc[:,1], ac[0,:], ac[1,:])
    else:
        f4 = ax1.quiver(CA3.neuron_model.pc[:,0], CA3.neuron_model.pc[:,1], ac[0,:], ac[1,:])
    
    ####### FIRING RATES #################

    colorbars.append(make_firing_rates_plot(ax4, firing_rate_store_AC[:,:,trial]))
    colorbars.append(make_firing_rates_plot(ax3, firing_rate_store_CA1[:,:,trial]))
    colorbars.append(make_firing_rates_plot(ax2, firing_rate_store_CA3[:,:,trial]))
    fig = ax4.get_figure()
    fig.subplots_adjust(hspace=0.2)

    ax4.set_yticks(list(range(len(AC.thetas))))
    ax4.set_yticklabels(action_degree)


    ############# WEIGHTS #################

    colorbars.append(make_weights_plot(ax5, CA1.SRM0_model.W, CA1.w_min, CA1.w_max))
    colorbars.append(make_action_weights_plot(ax6, AC.neuron_model.W, AC.w_min, AC.w_max))


    #ax[1,2].set_yticks(ticks)
    #ax[1,2].set_yticklabels(action_degree)

    for c in colorbars:
        c.ax.locator_params(nbins=5)

    fig.canvas.draw()
    plt.pause(0.01)

    F1.pop(0).remove()
    if ac is not None:
        f4.remove()
    for c in colorbars:
        c.remove()
