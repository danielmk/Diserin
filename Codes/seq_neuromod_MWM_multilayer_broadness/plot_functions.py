import matplotlib.pyplot as plt
import numpy as np


def initialize_plots(starting_position, r_goal, bounds_x, bounds_y,
                     CA1, offset_ca1, offset_ca3, CA3, c):
        
        plt.close()

        fig, ax = plt.subplots(ncols=4, nrows=2)
        fig.canvas.manager.full_screen_toggle() 
        fig.subplots_adjust(hspace = 0.5)

        ax[0,0].set_title('Trial 0')
        #Plot of reward places and initial position
        ax[0,0].plot(c[0]+r_goal*np.cos(np.linspace(-np.pi,np.pi,100)), c[1]+r_goal*np.sin(np.linspace(-np.pi,np.pi,100)),'b') #plot reward 1
        ax[0,0].plot(starting_position[0],starting_position[1], 'r',marker='o',markersize=5) #plot initial starting point
        #plot walls
        ax[0,0].plot([bounds_x[0],bounds_x[1]], [bounds_y[1],bounds_y[1]],c='k', ls='--',lw=0.5)
        ax[0,0].plot([bounds_x[0],bounds_x[1]], [bounds_y[0],bounds_y[0]],c='k', ls='--',lw=0.5)
        ax[0,0].plot([bounds_x[0],bounds_x[0]], [bounds_y[0],bounds_y[1]],c='k', ls='--',lw=0.5)
        ax[0,0].plot([bounds_x[1],bounds_x[1]], [bounds_y[0],bounds_y[1]],c='k', ls='--',lw=0.5)
        #plot cells
        ax[0,0].scatter(CA1.pc[:,0],CA1.pc[:,1], s=1, label="CA1")
        if offset_ca1!=offset_ca3:
            ax[0,0].scatter(CA3.pc[:,0],CA3.pc[:,1], s=1, label ="CA3")
            ax[0,0].legend()

        
        ax[0,1].set_title('CA3 firing rates')
        ax[0,2].set_title('CA1 firing rates')
        ax[0,3].set_title('Action Layer firing rates')
        
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
                 firing_rate_store, firing_rate_store_CA1,
                 firing_rate_store_CA3, w_tot, w_ca1,
                 CA1,ac):

    ax[0,0].set_title('Trial '+str(trial))
    #display trajectory of the agent in each trial
    trajectory = store_pos[trial]
    trajectory = trajectory[(trajectory[:,0]!=0.)|(trajectory[:,1]!=0.)]
    
    ax[0,0].plot(starting_position[0],starting_position[1],
                 'r',marker='o',markersize=5) #starting point
    f3 = ax[0,0].plot(trajectory[:, 0], trajectory[:,1]) #trajectory
    
    colorbars = []

    #display action neurons firing rates (activity bump)

    pos = ax[0,3].imshow(
        firing_rate_store_CA3[:,:,trial],origin='lower',cmap='Blues', interpolation='none',aspect='auto')
    #colorbar
    colorbars.append(fig.colorbar(pos, ax=ax[0,3]))

    pos = ax[0,2].imshow(
        firing_rate_store_CA1[:,:,trial],origin='lower',cmap='Blues', interpolation='none',aspect='auto')
    #colorbar
    colorbars.append(fig.colorbar(pos, ax=ax[0,2]))

    pos = ax[0,1].imshow(
        firing_rate_store[:,:,trial],origin='lower',cmap='Blues', interpolation='none',aspect='auto')
    #colorbar
    colorbars.append(fig.colorbar(pos, ax=ax[0,1]))
    
    # Display policy
    ac_norm=np.max(np.linalg.norm(ac,axis=0))
    f4 = ax[1,0].quiver(CA1.pc[:,0], CA1.pc[:,1], ac[0,:].T/ac_norm, ac[1,:].T/ac_norm)

    # Display weights
    pos = ax[1,1].imshow(w_ca1,cmap='Reds_r',origin='lower', interpolation='none',aspect='auto')
    colorbars.append(fig.colorbar(pos, ax=ax[1,1]))
    
    pos = ax[1,2].imshow(w_tot[:,0:CA1.N], cmap='Reds_r',origin='lower', interpolation='none',aspect='auto')
    colorbars.append(fig.colorbar(pos, ax=ax[1,2]))

    pos = ax[1,3].imshow(w_tot[:,CA1.N:],cmap='Reds_r',origin='lower', interpolation='none',aspect='auto')
    colorbars.append(fig.colorbar(pos, ax=ax[1,3]))

    plt.pause(0.00001)
    f3.pop(0).remove()
    f4.remove()
    for c in colorbars:
        c.remove()
