import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

def make_firing_rates_plot(fig, ax, rates):

    plot = ax.imshow(rates, origin='lower', cmap='Blues', interpolation='none',aspect='auto')

    cb = fig.colorbar(plot, ax=ax, format='%.1f')

    return cb

def make_weights_plot(fig, ax, w):

    plot = ax.imshow(w, origin='lower', cmap='Reds_r', interpolation='none',aspect='auto')

    cb = fig.colorbar(plot, ax=ax)

    return cb


def initialize_plots(r_goal, bounds_x, bounds_y,
                     CA1, offset_ca1, offset_ca3, CA3, c):
        
        plt.close()

        fig, ax = plt.subplots(ncols=4, nrows=2)
        fig.canvas.manager.full_screen_toggle() 
        fig.subplots_adjust(hspace = 0.5)

        #Plot of reward places and initial position
        ax[0,0].plot(c[0]+r_goal*np.cos(np.linspace(-np.pi,np.pi,100)), c[1]+r_goal*np.sin(np.linspace(-np.pi,np.pi,100)),'b') #plot reward 1
        #plot walls
        ax[0,0].plot([bounds_x[0],bounds_x[1]], [bounds_y[1],bounds_y[1]],c='k', ls='--',lw=0.5)
        ax[0,0].plot([bounds_x[0],bounds_x[1]], [bounds_y[0],bounds_y[0]],c='k', ls='--',lw=0.5)
        ax[0,0].plot([bounds_x[0],bounds_x[0]], [bounds_y[0],bounds_y[1]],c='k', ls='--',lw=0.5)
        ax[0,0].plot([bounds_x[1],bounds_x[1]], [bounds_y[0],bounds_y[1]],c='k', ls='--',lw=0.5)
        #plot cells
        ax[0,0].scatter(CA1.pc[:,0],CA1.pc[:,1], s=1, label="CA1")
        if offset_ca1!=offset_ca3:
            ax[0,0].scatter(CA3.pc[:,0],CA3.pc[:,1], s=1, label ="CA3")

        ax[0,3].set_title('CA3 firing rates')
        ax[0,2].set_title('CA1 firing rates')
        ax[0,1].set_title('Action Layer firing rates')

        ax[1,0].set_title('Agent''s policy')
        ax[1,1].set_title('CA3-CA1 weights')
        ax[1,2].set_title('CA1-Action weights')
        ax[1,3].set_title('Action Lateral weights')

        ax[1,0].set_xlim([CA1.pc.min(),CA1.pc.max()])
        ax[1,0].set_ylim([CA1.pc.min(),CA1.pc.max()])

        plt.pause(0.00001)

        return fig, ax


def update_plots(fig, ax,
                 trial, store_pos, starting_position,
                 firing_rate_store_AC, firing_rate_store_CA1,
                 firing_rate_store_CA3, w_tot, w_ca1,
                 CA1,ac, theta_actor):

    
    ticks = [4+4*i for i in range(9)]
    action_degree = [str(int(x)) for x in list(theta_actor[ticks]/(2*np.pi)*360)]

    ax[0,0].set_title('Trial '+str(trial))
    #display trajectory of the agent in each trial
    trajectory = store_pos[trial]
    trajectory = trajectory[(trajectory[:,0]!=0.)|(trajectory[:,1]!=0.)]
    
    ax[0,0].plot(starting_position[0],starting_position[1],
                 'r',marker='o',markersize=5) #starting point
    f3 = ax[0,0].plot(trajectory[:, 0], trajectory[:,1]) #trajectory
    
    colorbars = []

    cb = make_firing_rates_plot(fig, ax[0,1], firing_rate_store_AC[:,:,trial])
    colorbars.append(cb)
    ax[0,1].set_yticks(ticks)
    ax[0,1].set_yticklabels(action_degree)

    cb = make_firing_rates_plot(fig, ax[0,2], firing_rate_store_CA1[:,:,trial])
    colorbars.append(cb)

    cb = make_firing_rates_plot(fig, ax[0,3], firing_rate_store_CA3[:,:,trial])
    colorbars.append(cb)

    #ax[0,1].set_yticklabels(list(theta_actor/(2*np.pi)*360))
    
    # Display policy
    if ac is not None:
        ac_norm=np.max(np.linalg.norm(ac,axis=0))
        f4 = ax[1,0].quiver(CA1.pc[:,0], CA1.pc[:,1], ac[0,:].T/ac_norm, ac[1,:].T/ac_norm)

    # Display weights
    cb = make_weights_plot(fig, ax[1,1], w_ca1)
    colorbars.append(cb)

    cb = make_weights_plot(fig, ax[1,2], w_tot[:,0:CA1.N])
    colorbars.append(cb)
    ax[1,2].set_yticks(ticks)
    ax[1,2].set_yticklabels(action_degree)

    cb = make_weights_plot(fig, ax[1,3], w_tot[:,CA1.N:])
    colorbars.append(cb)
    ax[1,3].set_yticks(ticks)
    ax[1,3].set_yticklabels(action_degree)
    ax[1,3].set_xticks(ticks)
    ax[1,3].set_xticklabels(action_degree)


    for c in colorbars:
        c.ax.locator_params(nbins=5)

    plt.pause(0.00001)
    f3.pop(0).remove()
    if ac is not None:
        f4.remove()
    for c in colorbars:
        c.remove()
