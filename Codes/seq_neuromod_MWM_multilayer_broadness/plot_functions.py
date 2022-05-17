import matplotlib.pyplot as plt
import numpy as np


def initialize_plots(starting_position, r_goal, bounds_x, bounds_y,
                     CA1, offset_ca1, offset_ca3, CA3, c):
        
        plt.close()

        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6))= plt.subplots(ncols=3, nrows=2, figsize=(8, 8))
        fig.subplots_adjust(hspace = 0.5)

        #Plot of reward places and initial position
        ax1.plot(c[0]+r_goal*np.cos(np.linspace(-np.pi,np.pi,100)), c[1]+r_goal*np.sin(np.linspace(-np.pi,np.pi,100)),'b') #plot reward 1
        ax1.plot(starting_position[0],starting_position[1], 'r',marker='o',markersize=5) #plot initial starting point

        #plot walls
        ax1.plot([bounds_x[0],bounds_x[1]], [bounds_y[1],bounds_y[1]],c='k', ls='--',lw=0.5)
        ax1.plot([bounds_x[0],bounds_x[1]], [bounds_y[0],bounds_y[0]],c='k', ls='--',lw=0.5)
        ax1.plot([bounds_x[0],bounds_x[0]], [bounds_y[0],bounds_y[1]],c='k', ls='--',lw=0.5)
        ax1.plot([bounds_x[1],bounds_x[1]], [bounds_y[0],bounds_y[1]],c='k', ls='--',lw=0.5)


        ax1.scatter(CA1.pc[:,0],CA1.pc[:,1], s=1, label="CA1")

        if offset_ca1!=offset_ca3:
            ax1.scatter(CA3.pc[:,0],CA3.pc[:,1], s=1, label ="CA3")
            ax1.legend()

        ax1.set_title('Trial 0')
        ax2.set_title('Action neurons firing rates')
        ax3.set_title('CA1 firing rates')
        ax4.set_title('CA3 firing rates')
        ax5.set_title('Mean weights')
        ax6.set_title('Agent''s policy')

        ax6.set_xlim([CA1.pc.min(),CA1.pc.max()])
        ax6.set_ylim([CA1.pc.min(),CA1.pc.max()])

        plt.pause(0.00001)

        return fig, ax1, ax2, ax3, ax4 ,ax5, ax6


def update_plots(fig, ax1, ax2, ax3, ax4, ax5, ax6,
                 trial, store_pos, starting_position,
                 firing_rate_store, firing_rate_store_CA1,
                 firing_rate_store_CA3, w_tot,
                 CA1,ac):

    ax1.set_title('Trial '+str(trial))
    #display trajectory of the agent in each trial
    trajectory = store_pos[trial]
    trajectory = trajectory[(trajectory[:,0]!=0.)|(trajectory[:,1]!=0.)]
    f3 = ax1.plot(trajectory[:, 0], trajectory[:,1]) #trajectory
    ax1.plot(starting_position[0],starting_position[1],'r',marker='o',markersize=5) #starting point

    #display action neurons firing rates (activity bump)
    pos = ax2.imshow(
        firing_rate_store[:,:,trial],cmap='Blues', interpolation='none',aspect='auto')
    #colorbar
    if trial==0:
        fig.colorbar(pos, ax=ax2)

    pos = ax3.imshow(
        firing_rate_store_CA1[:,:,trial],cmap='Blues', interpolation='none',aspect='auto')
    #colorbar
    if trial==0:
        fig.colorbar(pos, ax=ax3)

    pos = ax4.imshow(
        firing_rate_store_CA3[:,:,trial],cmap='Blues', interpolation='none',aspect='auto')
    #colorbar
    if trial==0:
        fig.colorbar(pos, ax=ax4)
    
    
    
    #display weights over the open field, averaged over action neurons
    w_plot = np.mean(w_tot[:,0:CA1.N],axis=0) #use weights as they were at the beginning of the trial
    w_plot = np.reshape(w_plot,(int(np.sqrt(CA1.N)),int(np.sqrt(CA1.N))))
    pos2 = ax5.imshow(w_plot,cmap='Reds_r',origin='lower', interpolation='none',aspect='auto')
    #set(gca,'YDir','normal')
    if trial==1:
        fig.colorbar(pos2, ax=ax5)

    #plot policy as a vector field
    #filter zero values
    ac_norm=np.max(np.linalg.norm(ac,axis=0))
    f4=ax6.quiver(CA1.pc[:,0], CA1.pc[:,1], ac[0,:].T/ac_norm, ac[1,:].T/ac_norm)
    fig.canvas.draw()
    plt.pause(0.00001)
    f3.pop(0).remove()
    f4.remove()
    pos2.remove()
