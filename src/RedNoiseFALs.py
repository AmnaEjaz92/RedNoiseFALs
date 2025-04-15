import sys
import astropy
from astropy.timeseries import LombScargle
import scipy.optimize as sopt
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
plt.rcParams.update({"font.size":16, "axes.labelsize":16, "font.family":"sans-serif", "font.sans-serif":"Arial"})
    
    
#=====================================================================================================================

                                        # POWER SPECTRUM     

#======================================================================================================================    

def LSP(time,obs,fmax,m=3,plot=True,detrending=True):
    
        
    try: 
            arrays_expected = ((type(time) is np.ndarray) and (type(obs) is np.ndarray))
            if (not arrays_expected):
                raise ValueError
    except ValueError:
            print("Observation times and observations must be arrays")
            
            return                  
    
    try:
            test = (len(time) == len(obs))
            if (not test):
                raise ValueError
    except ValueError:
            print("Number of observation times must equal number of data points")
            return
    
    try:
            if fmax < 0:
                raise ValueError
    except ValueError:
            print("Nyquist frequency must be greater than 0 - no frequency grid calculated")
            return
    try:
            good_oversample = (((type(m) is int) or (type(m) is float)) \
                              and (m > 0))
            if not good_oversample:
                raise ValueError
    except ValueError:
            print("Oversample factor must be greater than 0 - no frequency grid calculated")
            return
        
     # Sorting
    indices = np.argsort(time)
    time = time[indices]
    obs = obs[indices]
    

    #Remove a linear trend
    if detrending:
        linear_trend = np.poly1d(np.polyfit(time, obs, 1))
        obs = obs - linear_trend(time)
    
    # Normalizing and mean subtraction
    std = np.std(obs)
    obs = (obs-np.mean(obs))/std
    
    
    
    #Setting up the frequency grid
    t = time[len(time)-1]-time[0] # length of time baseline 
    rr = 0.5*(1/t) ## 0.5R
    N_gridpoints = int((fmax/rr)*m)
    
    # Frequency grid
    fgrid = np.linspace(rr,fmax,num=N_gridpoints,endpoint=True)

    #LombScargle Periodogram
    ls = LombScargle(time,obs,normalization="psd")
    LS = ls.power(fgrid)
    false_alarm = ls.false_alarm_level([0.05, 0.01, 0.001], method = 'bootstrap')
    
    if plot:
        plt.figure(figsize=(14,8))
        plt.semilogy(fgrid,LS,color="green",alpha=0.6)
        plt.axhline(false_alarm[0], linestyle = 'dotted', color = 'r',label="5% FAP")
        plt.axhline(false_alarm[1], linestyle = 'dotted', color = 'g',label="1% FAP")
        plt.axhline(false_alarm[2], linestyle = 'dotted', color = 'k',label="0.1% FAP")
        plt.grid(axis="both")
        plt.xlabel(r"$f$ (days$^{-1}$)")
        plt.ylabel(r"$\hat{S}^{LS}(f)$")
        plt.title(r"Lomb-Scargle periodogram")
        plt.legend(loc='lower left',fontsize='small', ncol=2, facecolor='white', framealpha=1)


    return fgrid,LS,false_alarm




#=====================================================================================================================

                                        # POWER LAW FIT     

#======================================================================================================================    
def pl_fit(fgrid,LS,x0,Plot=True,Objective_plot=True,plot_limits=[(-3,0),(-3,3)],display_fitting_result=True,loglog_plot=True, method="Nelder-Mead",tol=10**-8):

    try: 
            arrays_expected = ((type(fgrid) is np.ndarray) and (type(LS) is np.ndarray))
            if (not arrays_expected):
                raise ValueError
    except ValueError:
            print("Frequency grids and LombScargle power must be arrays")
            
            return                  
    
    try:
            test = (len(fgrid) == len(LS))
            if (not test):
                raise ValueError
    except ValueError:
            print("The length of frequency grid should be equal to the power grid")
            return
    
### x0 needs to be a list with 2 elements for power law fit
    try:
            list_expected = ((type(x0) is list) and (len(x0) == 2))       
            if (not list_expected):
                raise ValueError
    except ValueError:
            print("The initial guess x0 should be a list with exactly 2 elements")
            return       
                   
                   
    # Take logs
    
    flog = np.log10(fgrid)
    plog = np.log10(LS)

    #Linear function
    
    def func(x,a,b):
            return a*x+b
        
    #Whittle likelihood function  
    
    def wnll_pl(params):
            spec = 10**func(flog, params[0],params[1])
            return sum((np.log(spec) + (LS/spec)))

    # Call_back function
    
    objective_values = []
    def callback_function(x):
        objective_value = wnll_pl(x)
        objective_values.append(objective_value)

    #Minimization
    estspec_wnll = minimize(wnll_pl, x0, method=method,tol=tol,callback=callback_function)    

    #Whittle_likelihood and parameters
    
    whittle_ll = estspec_wnll.fun
    slope = estspec_wnll.x[0]
    intercept = estspec_wnll.x[1]
    
 

    #Printing results
    if display_fitting_result:
        print("----------------------- POWER LAW FITTING RESULTS -------------------")
        print(estspec_wnll.message)
        print("Slope = %0.2f"%(estspec_wnll.x[0]))
        print("Intercept = %0.2f"%(estspec_wnll.x[1]))
        print("Whittle negative log-likelihood = %0.2f"%(estspec_wnll.fun))
 
   # Plot of the fit   

    if Plot==True:
        plt.figure(figsize=(9,6))
        if loglog_plot:
            plt.loglog(fgrid,LS,color="green",alpha=0.6)
            plt.loglog(fgrid,10**func(flog,estspec_wnll.x[0],estspec_wnll.x[1]),color="purple",label="Power law fit")
        else:
            plt.semilogy(fgrid,LS,color="green",alpha=0.6)
            plt.semilogy(fgrid,10**func(flog,estspec_wnll.x[0],estspec_wnll.x[1]),color="purple",label="Power law fit")

        plt.grid(axis="both")
        plt.xlabel(r"$f$ (days$^{-1}$)")
        plt.ylabel(r"$\hat{S}^{LS}(f)$")
        plt.title(r"Power law fit")
        plt.legend(loc='lower left',fontsize='small', ncol=2, facecolor='white', framealpha=1)

    # Minimization check plots
    
    if Objective_plot==True:

    #2D-grid and objective function

        plim = np.array(plot_limits)
        x = np.linspace(plim[0][0],plim[0][1],100)
        y = np.linspace(plim[1][0],plim[1][1],100)
        X,Y = np.meshgrid(x,y)

        ofunc = np.zeros((len(x),len(y)))

        for i in range(len(x)):
            for j in range(len(y)):
                z = np.array([X[i,j],Y[i,j]]) 
                ofunc[i,j] = wnll_pl(z)

        min_idx = np.unravel_index(np.argmin(ofunc), ofunc.shape)
        min_x = X[min_idx]
        min_y = Y[min_idx]
        
    #1D objective function

        param2= min_y
        obj_func1 = np.zeros(len(x))
        for i in range(len(x)):
            obj_func1[i]= wnll_pl([x[i],param2])
            
        param1= min_x
        obj_func2 = np.zeros(len(x))
        for i in range(len(x)):
            obj_func2[i]= wnll_pl([param1,y[i]])


        #Plotting the 4 plots together
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10,7))
        fig.suptitle("Minimization check - Objective function plots")
        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.4)

        # Plot 1 : Objective function vs. Iterations
        ax1.plot(objective_values)
        ax1.set_xlabel("No. of iterations")
        ax1.set_ylabel(r"$-\mathcal{L}(\theta)$")
        
        # Plot 2 : 2D plot of objective function vs. parameters
        levels = np.logspace(np.log10(np.min(ofunc)), np.log10(np.max(ofunc)),256)#np.linspace(np.min(ofunc),np.max(ofunc),256)
        norm = colors.BoundaryNorm(boundaries=levels,ncolors=256)
        ax2.pcolormesh(X,Y, ofunc,norm = norm)
        c0 = ax2.scatter(min_x, min_y, color='red', marker='o',s=10, label = r"$-\mathcal{L}(\theta)_{min}$")
        cb = fig.colorbar(c0, ax=ax2)
        cb.set_label(label=r"$-\mathcal{L}(\theta)$",weight='bold')
        ax2.set_xlabel(r"$p$")
        ax2.set_ylabel(r"a")
        ax2.legend(fontsize="small")

        # Plot 3 : Objective function vs. parameter 1
        ax3.plot(x,obj_func1)
        ax3.scatter(x[np.argmin(obj_func1)],np.min(obj_func1),color="black",label=r"$-\mathcal{L}(p)_{min}$")
        ax3.set_xlabel(r"$p$")
        ax3.set_ylabel(r"$-\mathcal{L}(p)$")
        ax3.legend(fontsize="small")
        
        # Plot 4 : Objective function vs. parameter 2
        ax4.plot(y,obj_func2)
        ax4.scatter(y[np.argmin(obj_func2)],np.min(obj_func2),color="black",label=r"$-\mathcal{L}(a)_{min}$")
        ax4.set_xlabel(r"$a$")
        ax4.set_ylabel(r"$-\mathcal{L}(a)$")
        ax4.legend(fontsize="small")
            
    
    return whittle_ll,slope,intercept,estspec_wnll

#=====================================================================================================================

                                                    #AR(1) fit      


#======================================================================================================================    

def ar1_fit(fgrid,LS,x0,Plot=True,Objective_plot=True,plot_limits=[(0.001,1),(0.5,1.5)],display_fitting_result=True, method="Nelder-Mead",tol=10**-8):

    try: 
            arrays_expected = ((type(fgrid) is np.ndarray) and (type(LS) is np.ndarray))
            if (not arrays_expected):
                raise ValueError
    except ValueError:
            print("Frequency grids and LombScargle power must be arrays")
            
            return                  
    
    try:
            test = (len(fgrid) == len(LS))
            if (not test):
                raise ValueError
    except ValueError:
            print("The length of frequency grid should be equal to the power grid")
            return
    
# x0 needs to be a 2D array for AR(1) with phi between 0 and 1             
                   
    try:
            list_expected = ((type(x0) is list) and (len(x0) == 2))       
            if (not list_expected):
                raise ValueError
    except ValueError:
            print("The initial guess x0 should be a list with exactly 2 elements")
            return   
                   
    if not (0 < x0[0]<= 1):   
                raise ValueError(r"The initial phi value must be between 0 and 1")                   
                return     
    # AR(1) function 
    def ar1(frequency, phi, sigma):
            return (sigma**2)/(1-2*phi*np.cos(2*np.pi*frequency)+phi**2)

    #Whittle likelihood
    def wnll_ar1(params):
                spec = ar1(fgrid, params[0],params[1])
                return sum((np.log(spec) + (LS/spec)))
            
    #Callback function
    objective_values = []
    def callback_function(x):
        objective_value = wnll_ar1(x)
        objective_values.append(objective_value)
            
    # Minimization             
    bnds = ((-1, 0.99999999), (0, None)) #Changed it from 1 to avoid getting 1 for KIC shorter grid
    estspec_wnll = minimize(wnll_ar1, x0, method=method, tol=tol, bounds = bnds,callback=callback_function)

    #Whittle likelihood and parameters
    whittle_ll = estspec_wnll.fun
    phi = estspec_wnll.x[0]
    sigma = estspec_wnll.x[1]

    
    #Printing results
    if display_fitting_result:
        print("----------------------- AR(1) FITTING RESULTS -------------------")
        print(estspec_wnll.message)
        print("Phi = %0.2f"%(estspec_wnll.x[0]))
        print("Sigma = %0.2f"%(estspec_wnll.x[1]))
        print("Whittle negative log-likelihood = %0.2f"%(estspec_wnll.fun))


    #Plotting the fits
    
    if Plot == True:
        plt.figure(figsize=(10,6))
        plt.semilogy(fgrid,LS,color="green",alpha=0.6)
        plt.semilogy(fgrid,ar1(fgrid,estspec_wnll.x[0],estspec_wnll.x[1]),color="purple",label="AR(1) fit")
        plt.grid(axis="both")
        plt.xlabel(r"$f$ (days$^{-1}$)")
        plt.ylabel(r"$\hat{S}^{LS}(f)$")
        plt.title(r"AR(1) fit")
        plt.legend(loc='lower left',fontsize='small', ncol=2, facecolor='white', framealpha=1)

        
    #Minimization checks plot
    if Objective_plot==True:
        
    #2D grid and objective function
    
        plim = np.array(plot_limits)
        x = np.linspace(plim[0][0],plim[0][1],100)
        y = np.linspace(plim[1][0],plim[1][1],100)
        X,Y = np.meshgrid(x,y)

        ofunc = np.zeros((len(x),len(y)))

        for i in range(len(x)):
            for j in range(len(y)):
                z = np.array([X[i,j],Y[i,j]]) 
                ofunc[i,j] = wnll_ar1(z)

        min_idx = np.unravel_index(np.argmin(ofunc), ofunc.shape)
        min_x = X[min_idx]
        min_y = Y[min_idx]
        
        #1D objective function and parameters

        param2= min_y
        obj_func1 = np.zeros(len(x))
        for i in range(len(x)):
            obj_func1[i]= wnll_ar1([x[i],param2])
            
        param1= min_x
        obj_func2 = np.zeros(len(x))
        for i in range(len(x)):
            obj_func2[i]= wnll_ar1([param1,y[i]])

        #Plotting the 4 plots together
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10,7))
        fig.suptitle("Minimization check - Objective function plots")
        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.4)
        
        # Plot 1: Objective function vs. iterations
        ax1.plot(objective_values)
        ax1.set_xlabel("No. of iterations")
        ax1.set_ylabel(r"$-\mathcal{L}(\theta)$")
        
        # Plot 2: 2D objective function vs. parameters
        levels = np.logspace(np.log10(np.min(ofunc)), np.log10(np.max(ofunc)),20)#np.linspace(np.min(ofunc),np.max(ofunc),256)
        norm = colors.BoundaryNorm(boundaries=levels,ncolors=256)
        ax2.pcolormesh(X,Y, ofunc,norm = norm)
        c0 = ax2.scatter(min_x, min_y, color='red', marker='o',s=10,label = r"$-\mathcal{L}(\theta)_{min}$")
        cb = fig.colorbar(c0, ax=ax2)
        cb.set_label(label=r"$-\mathcal{L}(\theta)$",weight='bold')
        ax2.set_xlabel(r"$\phi$")
        ax2.set_ylabel(r"$\sigma$")
        ax2.legend(fontsize="small")

        # Plot 3: Objective function vs. parameter 1
        ax3.plot(x,obj_func1)
        ax3.scatter(x[np.argmin(obj_func1)],np.min(obj_func1),color="black",label=r"$-\mathcal{L}(\phi)_{min}$")
        ax3.set_xlabel(r"$\phi$")
        ax3.set_ylabel(r"$-\mathcal{L}(\phi)$")
        ax3.legend(fontsize="small")

        # Plot 4: Objective function vs. parameter 2
        ax4.plot(y,obj_func2)
        ax4.scatter(y[np.argmin(obj_func2)],np.min(obj_func2),color="black",label=r"$-\mathcal{L}(\sigma)_{min}$")
        ax4.set_xlabel(r"$\sigma$")
        ax4.set_ylabel(r"$-\mathcal{L}(\sigma)$")
        ax4.legend(fontsize="small")

        
 
        
    return whittle_ll,phi,sigma,estspec_wnll

#=====================================================================================================================

                                                #White noise fit


#======================================================================================================================    

def wn_fit(fgrid,LS,x0,Plot=True,Objective_plot=True,plot_limits=[(0.5,5)],display_fitting_result=True,method="Nelder-Mead",tol=10**-8):

    try: 
            arrays_expected = ((type(fgrid) is np.ndarray) and (type(LS) is np.ndarray))
            if (not arrays_expected):
                raise ValueError
    except ValueError:
            print("Frequency grids and LombScargle power must be arrays")
            
            return                  
    
    try:
            test = (len(fgrid) == len(LS))
            if (not test):
                raise ValueError
    except ValueError:
            print("The length of frequency grid should be equal to the power grid")
            return
                   
    try:
            float_expected = ((type(x0) is float))       
            if (not float_expected):
                raise ValueError
    except ValueError:
            print("The initial guess x0 should be a real number")
            return   
              
                   
                   
    # power function
    def func_wn(x,a):
            return a
 
    # Whittle likelihood function 
    def wnll_wn(a):
            spec = func_wn(fgrid, a)
            return sum((np.log(spec)+(LS/spec)))
    
    # Callback function
    objective_values = []
    def callback_function(x):
        objective_value = wnll_wn(x)
        objective_values.append(objective_value)
                 
    # Minimization
    estspec_wn = minimize(wnll_wn, x0, method=method, tol=tol,callback=callback_function)

    # Whittle likelihood and power
    whittle_ll = estspec_wn.fun
    power = estspec_wn.x[0]

    #Printing the results
    if display_fitting_result:
        print("----------------------- WHITE NOISE FITTING RESULTS -------------------")
        print(estspec_wn.message)
        print("Power of white noise = %0.2f"%(estspec_wn.x[0]))
        print("Whittle negative log-likelihood = %0.2f"%(estspec_wn.fun))

    #Plotting the fit
    if Plot== True:
        plt.figure(figsize=(10,6))
        
        plt.semilogy(fgrid,LS,color="green",alpha=0.6)
        plt.axhline(estspec_wn.x[0],linestyle="--",color="red",label="White noise fit")
        plt.grid(axis="both")
        plt.xlabel(r"$f$ (days$^{-1}$)")
        plt.ylabel(r"$\hat{S}^{LS}(f)$")
        plt.title(r"White noise fit")
        plt.legend(loc='lower left',fontsize='small', ncol=2, facecolor='white', framealpha=1)

        
    # Minimization checks    
    if Objective_plot==True:

        # Objective function and grid
        plim = np.array(plot_limits)
        x = np.linspace(plim[0][0],plim[0][1],100)
        obj_func1 = np.zeros(len(x))
        for i in range(len(x)):
            obj_func1[i]= wnll_wn(x[i])

        # Plotting objective function    
        fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10,3))
        fig.suptitle('Minimization check - Objective function plots')
        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.8, wspace=0.4, hspace=0.4)

        # Plot 1: Objective function vs. iteration
        ax1.plot(objective_values)
        ax1.set_xlabel("No. of iterations")
        ax1.set_ylabel(r"$-\mathcal{L}(c)$")
        
        #Plot 2: Objective function vs. power
        ax2.plot(x,obj_func1)
        ax2.scatter(x[np.argmin(obj_func1)],np.min(obj_func1),color="black",label=r"$-\mathcal{L}(c)_{min}$")
        ax2.set_xlabel("$c$")
        ax2.set_ylabel(r"$-\mathcal{L}(c)$")

        
    return whittle_ll,power,estspec_wn

#=====================================================================================================================

                        #DISTRIBUTIONS OF PARAMETERS, WHITTLE LIKELIHOODS AND RMSE

#=====================================================================================================================

def gen_distributions(time,obs,eobs,fgrid,n_bootstrap=10000,x0_pl=[-0.5,-0.9],x0_ar1=[0.7,0.5],x0_wn=0.1,histograms=True,detrending=True,save_file="fitting_results.txt",output_file=True):

### time obs eobs fgrid arrays
    try: 
            arrays_expected = ((type(time) is np.ndarray) and (type(obs) is np.ndarray) and (type(fgrid) is np.ndarray) and (type(eobs) is np.ndarray))
            if (not arrays_expected):
                raise ValueError
    except ValueError:
            print("Observation times, observations, error bars and the frequency grid should be an array")
            
            return                                         
              
### arrays lengths
    try:
            test = (len(time) == len(obs)==len(eobs))
            if (not test):
                raise ValueError
    except ValueError:
            print("The length of observation times, observations and error bar arrays should be equal")
            return

    try:
            float_expected = ((type(x0_wn) is float))       
            if (not float_expected):
                raise ValueError
    except ValueError:
            print("The initial guess 'x0_wn' should be a real number")
            return

    try:
            list_expected = ((type(x0_ar1) is list) and (len(x0_ar1) == 2))       
            if (not list_expected):
                raise ValueError
    except ValueError:
            print("The initial guess 'x0_ar1' should be a list with exactly 2 elements")
            return
        
    if not (0 < x0_ar1[0]<= 1):   
                raise ValueError(r"The initial phi value must be between 0 and 1")                   
                return
        
    try:
            list_expected = ((type(x0_pl) is list) and (len(x0_pl) == 2))       
            if (not list_expected):
                raise ValueError
    except ValueError:
            print("The initial guess 'x0_pl' should be a list with exactly 2 elements")
            return                                      
              
    try:
            integer_expected = ((type(n_bootstrap) is int) and (0 < n_bootstrap))       
            if (not float_expected):
                raise ValueError
    except ValueError:
            print("The number of bootstrap 'n_bootstrap' should be an integer greater than zero")
            return

            
            
    #sorting
    ind = np.argsort(time)
    time = time[ind]
    obs = obs[ind]-np.mean(obs[ind])
    eobs = eobs[ind]
    #Remove a linear trend
    if detrending:
        linear_trend = np.poly1d(np.polyfit(time, obs, 1))
        obs = obs - linear_trend(time)

    #Normalizing
    std = np.std(obs)
    obs= obs/std
    eobs = eobs/std

    wnll_ar1_dist = np.zeros((n_bootstrap))
    wnll_wn_dist = np.zeros((n_bootstrap))
    wnll_pl_dist = np.zeros((n_bootstrap))
    
    slope = np.zeros((n_bootstrap))
    intercept =  np.zeros((n_bootstrap))
    
    phi = np.zeros((n_bootstrap))
    sigma =  np.zeros((n_bootstrap))
    
    N = len(time)
    
    try:
        if os.path.exists(save_file):
            raise FileExistsError(f"The file '{save_file}' already exists.")
    except FileExistsError as e:
        print(e)
        return
    
    if output_file:
        g = open(save_file,'w')
        g.write('%s %s %s %s %s %s %s %s \n'%("WNLL_PL","Slope","Intercept","WNLL_AR1","Phi","Sigma","WNLL_WN","Power"))
        g.close()

    ls_real = LombScargle(time,obs,normalization="psd")
    LS_real = ls_real.power(fgrid)    
    
    for k in range(n_bootstrap):  
  
        rand_num = np.random.randn(N)
        new_obs = np.zeros((N))
        deviation_n = rand_num*eobs
        new_obs = deviation_n + obs

        #LombScargle Periodogram
        ls = LombScargle(time,new_obs,normalization="psd")
        LS = ls.power(fgrid)
        
        pl_out = pl_fit(fgrid,LS,x0_pl,Plot=False,Objective_plot=False,display_fitting_result=False)
        ar1_out = ar1_fit(fgrid,LS,x0_ar1,Plot=False,Objective_plot=False,display_fitting_result=False)
        wn_out = wn_fit(fgrid,LS,x0_wn,Plot=False,Objective_plot=False,display_fitting_result=False) 
        
        wnll_pl_dist[k] = pl_out[0]
        wnll_ar1_dist[k] = ar1_out[0]
        wnll_wn_dist[k] = wn_out[0]
        
        slope[k] = pl_out[1]
        intercept[k] = pl_out[2]
        
                
        phi[k] = ar1_out[1]
        sigma[k] = ar1_out[2]

        if output_file:
            f = open(save_file,"a")
            f.write(f"{pl_out[0]:.{20}f} {pl_out[1]:.{20}f} {pl_out[2]:.{20}f} {ar1_out[0]:.{20}f}  {ar1_out[1]:.{20}f}  {ar1_out[2]:.{20}f}  {wn_out[0]:.{20}f}  {wn_out[1]:.{20}f} \n")
    
            f.close()
        
       
    if histograms:
        bins = 100
        plt.figure(figsize=(10,6))
        plt.title("Whittle likelihood distributions")
        plt.hist(wnll_pl_dist,label="Power law",alpha =0.5,bins=bins)
        plt.hist(wnll_ar1_dist,label="AR(1)",alpha =0.5,bins=bins)
        plt.hist(wnll_wn_dist,label="White Noise",alpha =0.5,bins=bins)
        plt.legend()


    return wnll_pl_dist,wnll_ar1_dist,wnll_wn_dist,slope,intercept,phi,sigma


#=====================================================================================================================

                                        #FAPS FROM AR(1)

#=====================================================================================================================

                   
                   
def fal_ar1(time,obs,fgrid,phi,sigma,n_bootstrap=10000,detrending=True,Plot=True,title="FALs based on AR(1)"):
    
    try: 
            arrays_expected = ((type(time) is np.ndarray) and (type(obs) is np.ndarray) and (type(fgrid) is np.ndarray))
            if(not arrays_expected):
                raise ValueError
    except ValueError:
            print("Observation times, observations, and frequency grid should be arrays")
            
            return                                         
    try:
            test = (len(time) == len(obs))
            if (not test):
                raise ValueError
    except ValueError:
            print("The length of observation times and observations arrays should be equal")
            return
                               
    if ((type(phi) is np.ndarray) and (type(sigma) is np.ndarray)):
        try:
                test = (len(phi) == len(sigma)==n_bootstrap)
                if (not test):
                    raise ValueError
        except ValueError:
                print("The length of distributions of phi and sigma should be equal to number of bootstraps")
                return


    
    # Sorting
    indices = np.argsort(time)
    time = time[indices]
    obs = obs[indices]-np.mean(obs[indices]) # Mean subtraction

    #Remove a linear trend
    if detrending:
        linear_trend = np.poly1d(np.polyfit(time, obs, 1))
        obs = obs - linear_trend(time)
    
    # Normalizing
    std = np.std(obs)
    obs = obs/std
    
    ls_real = LombScargle(time,obs,normalization="psd")
    LS_real = ls_real.power(fgrid)
    fal_white = ls_real.false_alarm_level([0.05, 0.01,0.001], method = 'bootstrap')
        
    N_gridpoints = len(fgrid) # No. of gridpoints in frequency grid
    n = len(time)             # No. of data points in timeseries
    
    new_obs = np.zeros((n)) # Simulated timeseries array
    spec = np.zeros((N_gridpoints,n_bootstrap)) #Save the power spectrum for multiple timeseries
    
    rand_int = np.random.randn(n_bootstrap)
    

    # Simulating the fake timeseries based on red noise using AR(1) parameters 

    for j in range(n_bootstrap):
        
        if type(phi) == float:
                      phi[j] = phi
        if type(sigma) == float:
                      sigma[j] = sigma
                           
        
        tau = -1/(np.log(np.abs(phi[j]))) # Persistence time
        new_obs[0] = rand_int[j]
        epsilon = sigma[j]*np.random.randn(n)
        
        for i in range(1,n):
            delta_t = time[i]-time[i-1]
            new_obs[i] = new_obs[i-1]*np.exp(-delta_t/tau) + epsilon[i]

# Normalizing 
        std_obs = np.std(new_obs)
        Obs = new_obs/std_obs

# Generating the power spectrum for each timeseries and saving the powers across all frequency grid
        ls = LombScargle(time,Obs,normalization="psd")
        LS = ls.power(fgrid)
        spec[:,j] = LS
        
# Calculating the percentiles to get the False Alarm Probabilities       
    percentiles = np.zeros((N_gridpoints,4))
    for f in range(N_gridpoints):
        percentiles[f,:] = np.percentile(spec[f,:],[95.0,99.0,99.9,50.0])
    
    if Plot:
        plt.figure(figsize=(10,6))
        plt.title(title)
        plt.xlabel(r"$f$ (days$^{-1}$)")
        plt.ylabel(r"$\hat{S}^{LS}(f)$")
        plt.semilogy(fgrid,percentiles[:,0],color="red",label="5% Red Noise FAL")
        plt.semilogy(fgrid,percentiles[:,1],color="orange",label="1% Red Noise FAL")
        plt.semilogy(fgrid,percentiles[:,2],color="dodgerblue",label="0.1% Red Noise FAL")
        plt.axhline(fal_white[1], linestyle = '--', color = 'black',label="1% White Noise FAL")
        plt.semilogy(fgrid,percentiles[:,3],color="purple", label = r"Bootstrap $50^{th}$ percentile")
        plt.semilogy(fgrid,LS_real,color="green",alpha=0.7)
        plt.grid(axis="both")
        plt.yticks()
        plt.xticks()   
        plt.legend(fontsize='small', ncol=1, facecolor='white', framealpha=1,bbox_to_anchor=(1.0, 1.0))

    
    return percentiles

#=====================================================================================================================

                                        #FAPS FROM POWER LAW

#=====================================================================================================================

def fal_pl(time,obs,fgrid,slope,intercept,n_bootstrap=10000,detrending=True,Plot=True,title="FALs based on power law"):

    try: 
            arrays_expected = ((type(time) is np.ndarray) and (type(obs) is np.ndarray) and (type(fgrid) is np.ndarray))
            if (not arrays_expected):
                raise ValueError
    except ValueError:
            print("Observation times, observations, and frequency grid should be arrays")
            
            return                                         
    try:
            test = (len(time) == len(obs))
            if (not test):
                raise ValueError
    except ValueError:
            print("The length of observation times and observations arrays should be equal")
            return
                                 
                               
                               
    if ((type(slope) is np.ndarray) and (type(intercept) is np.ndarray)):
        try:
                test = (len(slope) == len(intercept)==n_bootstrap)
                if (not test):
                    raise ValueError
        except ValueError:
                print("The length of distributions of slope and intercept should be equal to number of bootstraps")
                return
    
    # Sorting
    indices = np.argsort(time)
    time = time[indices]
    obs = obs[indices]-np.mean(obs[indices]) # Mean subtraction

    #Remove a linear trend
    if detrending:
        linear_trend = np.poly1d(np.polyfit(time, obs, 1))
        obs = obs - linear_trend(time)

    
    # Normalizing
    std = np.std(obs)
    obs = obs/std
    
    ls_real = LombScargle(time,obs,normalization="psd")
    LS_real = ls_real.power(fgrid)
    fal_white = ls_real.false_alarm_level([0.05, 0.01,0.001], method = 'bootstrap')
    
    flog = np.log10(fgrid)

# Linear function    
    def func(x,a,b):
            return a*x+b
        
    n = len(time)
    k = len(fgrid)
    spec = np.zeros((k,n_bootstrap))
    rand_int = np.random.randn(n_bootstrap)
        
    
    time = time-time[0]
    delta_w = 2*np.pi*(fgrid[1]-fgrid[0])
    
    for j in range(n_bootstrap):
                               
        if type(slope) == float:
                      slope[j] = slope
                               
        if type(intercept) == float:
                     intercept[j] = intercept
        
        #Power spectrum from power law
        ps_pl = 10**func(flog,slope[j],intercept[j])

        phi_k = np.random.rand(k)*2*np.pi
        new_obs = np.zeros((n))
        new_obs[0] = rand_int[j]
        
# Simulating the random new timeseries based on power law
        for i in range(1,n):
            t_i = time[i]
            new_obs[i] = np.sum(np.sqrt(ps_pl*delta_w)*np.cos(2*np.pi*fgrid*t_i + phi_k))

# Normalizing
        new_obs = new_obs/np.std(new_obs)

# Generating the power spectrum for each timeseries and saving the powers across all frequency grid

        ls_fts = LombScargle(time,new_obs,normalization="psd")
        LS_fts = ls_fts.power(fgrid)
        spec[:,j] = LS_fts

# Calculating the percentiles to get the False Alarm Probabilities               
    percentiles = np.zeros((k,4))
    for f in range(k):
        percentiles[f,:] = np.percentile(spec[f,:],[95.0,99.0,99.9,50.0]) 
    if Plot:
        plt.figure(figsize=(10,6))
        plt.title(title)
        plt.xlabel(r"$f$ (days$^{-1}$)")
        plt.ylabel(r"$\hat{S}^{LS}(f)$")
        plt.semilogy(fgrid,LS_real,color="green",alpha=0.7)
        plt.semilogy(fgrid,percentiles[:,0],color="red",label="5% Red Noise FAL")
        plt.semilogy(fgrid,percentiles[:,1],color="orange",label="1% Red Noise FAL")
        plt.semilogy(fgrid,percentiles[:,2],color="dodgerblue",label="0.1% Red Noise FAL")
        plt.semilogy(fgrid,percentiles[:,3],color="purple", label = r"Bootstrap $50^{th}$ percentile")
        plt.axhline(fal_white[1], linestyle = '--', color = 'black',label="1% White Noise FAL")
        plt.grid(axis="both")
        plt.yticks()
        plt.xticks()   
        plt.legend(fontsize='small', ncol=1, facecolor='white', framealpha=1,bbox_to_anchor=(1.0, 1.0))
    return percentiles

