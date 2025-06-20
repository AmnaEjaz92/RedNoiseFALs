import sys
import astropy
from astropy.timeseries import LombScargle
import scipy.optimize as sopt
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
import copy

plt.rcParams.update({"font.size":20, "axes.labelsize":20, "font.family":"sans-serif", "font.sans-serif":"Arial"})
    
    
#=====================================================================================================================

                                        # POWER SPECTRUM     

#======================================================================================================================    

class RedNoiseFALs:
    
    # *** Constructor***
    def __init__(self, time, obs,eobs,fmax,fmin="Default",detrending=True,plot=True):
        
        # time: vector of observation times
        # obs: vector of observed data (i.e. RV, S-index, H-alpha)
        # fmax: The maximum frequency to be observed
        # fmin: The minimum frequency of the grid if "Default" then fmin = R the Rayleigh resolution of the data set
        # detrending: If "True"(default) removes a linear trend from the data before computing LSP 
        # plot: If "True"(default) plots the LombScargle Periodogram along with the bootstrapped white noise FALs
        
        try:
            test = (len(time) == len(obs))
            if (not test):
                raise ValueError
        except ValueError:
            print("Number of observation times must equal number of data points")
            print("No RedNoiseFALs object created")
            return
        
       
         # Sorting
        indices = np.argsort(time)
        self.time = time[indices]
        self.obs = obs[indices]
        self.eobs = eobs[indices]
        
        # Saving the maximum and minimum frequencies 
        self.fnyq = fmax
        self.fmin = fmin
        
        # Removing a linear trend
        if detrending:
            linear_trend = np.poly1d(np.polyfit(self.time, self.obs, 1))
            self.obs = self.obs - linear_trend(self.time)

        # Normalizing and mean subtraction
        std = np.std(self.obs)
        self.obs = (self.obs-np.mean(self.obs))/std
        self.eobs = self.eobs/std

        # Length of the time series
        t = self.time[len(self.time)-1]-self.time[0] 

        #Setting up the frequency grid
        
        rr = (1/t) ## Rayleigh resolution
        N_gridpoints = int((self.fnyq/rr))

        # Frequency grid with spacing equal to one Rayleigh resolution
        self.fgrid = np.linspace(rr,self.fnyq,num=N_gridpoints,endpoint=True)
        
        # If the minimum frequency required by the user is not default one, find the grid point closest to the required minimum frequency and cut the grid at that point, to get the desired minimum frequency.
        if self.fmin!= "Default":   
            def find_index_closest_to_value(arr,target_value,tolerance):
                differences = np.abs(arr - target_value)
                min_index = np.argmin(differences)
                if differences[min_index] <= tolerance:
                    return min_index
                else:
                    raise ValueError('No grid point found near the specified fmin within the tolerence range')
            # Cutting the grid to start at the desired fmin        
            self.fgrid = self.fgrid[find_index_closest_to_value(self.fgrid,self.fmin,10**-4):]

        # Rayleigh grid-spaced LombScargle Periodogram
        ls = LombScargle(self.time,self.obs,normalization="psd")
        self.LS = ls.power(self.fgrid)
        
        # White noise fals
        self.fals = ls.false_alarm_level([0.05, 0.01, 0.001], method = 'bootstrap')

        if plot:

            plt.figure(figsize=(10,6))
            plt.semilogy(self.fgrid,self.LS,color="green",alpha=0.6)
            plt.axhline(self.fals[0], linestyle = 'dotted', color = 'r',label="5% FAP")
            plt.axhline(self.fals[1], linestyle = 'dotted', color = 'g',label="1% FAP")
            plt.axhline(self.fals[2], linestyle = 'dotted', color = 'k',label="0.1% FAP")
            plt.grid(axis="both")
            plt.xlabel(r"$f$ (days$^{-1}$)")
            plt.ylabel(r"$\hat{S}^{LS}(f)$")
            plt.title(r"Lomb-Scargle periodogram")
            plt.legend(loc='best',fontsize='small', ncol=2, facecolor='white', framealpha=1)



    #=====================================================================================================================

                                            # POWER LAW FIT     

    #======================================================================================================================    
    def pl_fit(self,x0,Plot=True,Objective_plot=True,plot_limits=[(-3,0),(-3,3)],display_fitting_result=True,loglog_plot=True, method="Nelder-Mead",tol=10**-8):

       
    ### x0 needs to be a list with 2 elements for power law fit
        try:
                list_expected = ((type(x0) is list) and (len(x0) == 2))       
                if (not list_expected):
                    raise ValueError
        except ValueError:
                print("The initial guess x0 should be a list with exactly 2 elements")
                return       

        # Saving the initial guess, method and tolerance for the minimization      
        self.x0_pl = x0
        self.method_pl = method
        self.tol_pl = tol

        # Take logs

        flog = np.log10(self.fgrid)
        plog = np.log10(self.LS)

        #Linear function

        def func(x,a,b):
                return a*x+b

        #Whittle likelihood function  

        def wnll_pl(params):
                spec = 10**func(flog, params[0],params[1])
                return sum((np.log(spec) + (self.LS/spec)))

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
            plt.figure(figsize=(10,6))
            if loglog_plot:
                plt.loglog(self.fgrid,self.LS,color="green",alpha=0.6)
                plt.loglog(self.fgrid,10**func(flog,estspec_wnll.x[0],estspec_wnll.x[1]),color="purple",label="Power law fit")
            else:
                plt.semilogy(self.fgrid,self.LS,color="green",alpha=0.6)
                plt.semilogy(self.fgrid,10**func(flog,estspec_wnll.x[0],estspec_wnll.x[1]),color="purple",label="Power law fit")

            plt.grid(axis="both")
            plt.xlabel(r"$f$ (days$^{-1}$)")
            plt.ylabel(r"$\hat{S}^{LS}(f)$")
            plt.title(r"Power law fit")
            plt.legend(loc='best',fontsize='small', ncol=2, facecolor='white', framealpha=1)

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

    def ar1_fit(self,x0,Plot=True,Objective_plot=True,plot_limits=[(0.001,1),(0.5,1.5)],display_fitting_result=True, method="Nelder-Mead",tol=10**-8):

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

        self.x0_ar1 = x0
        self.method_ar1 = method
        self.tol_ar1 = tol


                
        # AR(1) function 
        def ar1(frequency, phi, sigma):
                return (sigma**2)/(1-2*phi*np.cos(2*np.pi*frequency)+phi**2)

        #Whittle likelihood
        def wnll_ar1(params):
                    spec = ar1(self.fgrid, params[0],params[1])
                    return sum((np.log(spec) + (self.LS/spec)))

        #Callback function
        objective_values = []
        def callback_function(x):
            objective_value = wnll_ar1(x)
            objective_values.append(objective_value)

        # Minimization             
        bnds = ((0, 0.99999999), (0, None))
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
            plt.semilogy(self.fgrid,self.LS,color="green",alpha=0.6)
            plt.semilogy(self.fgrid,ar1(self.fgrid,estspec_wnll.x[0],estspec_wnll.x[1]),color="purple",label="AR(1) fit")
            plt.grid(axis="both")
            plt.xlabel(r"$f$ (days$^{-1}$)")
            plt.ylabel(r"$\hat{S}^{LS}(f)$")
            plt.title(r"AR(1) fit")
            plt.legend(loc='best',fontsize='small', ncol=2, facecolor='white', framealpha=1)


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

    def wn_fit(self,x0,Plot=True,Objective_plot=True,plot_limits=[(0.5,5)],display_fitting_result=True,method="Nelder-Mead",tol=10**-8):


        try:
                float_expected = ((type(x0) is float))       
                if (not float_expected):
                    raise ValueError
        except ValueError:
                print("The initial guess x0 should be a real number")
                return   

        self.x0_wn = x0
        self.method_wn = method
        self.tol_wn = tol


        # power function
        def func_wn(x,a):
                return a

        # Whittle likelihood function 
        def wnll_wn(a):
                spec = func_wn(self.fgrid, a)
                return sum((np.log(spec)+(self.LS/spec)))

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

            plt.semilogy(self.fgrid,self.LS,color="green",alpha=0.6)
            plt.axhline(estspec_wn.x[0],linestyle="--",color="red",label="White noise fit")
            plt.grid(axis="both")
            plt.xlabel(r"$f$ (days$^{-1}$)")
            plt.ylabel(r"$\hat{S}^{LS}(f)$")
            plt.title(r"White noise fit")
            plt.legend(loc='best',fontsize='small', ncol=2, facecolor='white', framealpha=1)


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

    def gen_distributions(self,n_bootstrap=10000,histograms=True,detrending=True,save_file="fitting_results.txt",save_output=True):

   


        wnll_ar1_dist = np.zeros((n_bootstrap))
        wnll_wn_dist = np.zeros((n_bootstrap))
        wnll_pl_dist = np.zeros((n_bootstrap))

        slope = np.zeros((n_bootstrap))
        intercept =  np.zeros((n_bootstrap))

        phi = np.zeros((n_bootstrap))
        sigma =  np.zeros((n_bootstrap))

        N_gridpoints =len(self.fgrid)
        N = len(self.time)

        try:
            if os.path.exists(save_file):
                raise FileExistsError(f"The file '{save_file}' already exists.")
        except FileExistsError as e:
            print(e)
            return

        if save_output:
            g = open(save_file,'w')
            g.write('%s %s %s %s %s %s %s %s \n'%("WNLL_PL","Slope","Intercept","WNLL_AR1","Phi","Sigma","WNLL_WN","Power"))
            g.close()
            
            
        old_obs = copy.deepcopy(self.obs)
        old_LS = copy.deepcopy(self.LS)
        
        for k in range(n_bootstrap):  

            rand_num = np.random.randn(N)
            new_obs = np.zeros((N))
            deviation_n = rand_num*self.eobs
            self.obs = deviation_n + self.obs
            
            #LombScargle Periodogram
            ls_new = LombScargle(self.time,self.obs,normalization="psd")
            self.LS = ls_new.power(self.fgrid)
            
            pl_output = self.pl_fit(self.x0_pl,method=self.method_pl,tol=self.tol_pl,Plot=False,Objective_plot=False,display_fitting_result=False)
            ar1_output = self.ar1_fit(self.x0_ar1,method=self.method_ar1,tol=self.tol_ar1,Plot=False,Objective_plot=False,display_fitting_result=False)
            wn_output = self.wn_fit(self.x0_wn,method=self.method_wn,tol=self.tol_wn,Plot=False,Objective_plot=False,display_fitting_result=False) 
            
            
            

            wnll_pl_dist[k] = pl_output[0]
            wnll_ar1_dist[k] = ar1_output[0]
            wnll_wn_dist[k] = wn_output[0]

            slope[k] = pl_output[1]
            intercept[k] = pl_output[2]


            phi[k] = ar1_output[1]
            sigma[k] = ar1_output[2]

            if save_output:
                f = open(save_file,"a")
                f.write(f"{pl_output[0]:.{20}f} {pl_output[1]:.{20}f} {pl_output[2]:.{20}f} {ar1_output[0]:.{20}f}  {ar1_output[1]:.{20}f}  {ar1_output[2]:.{20}f}  {wn_output[0]:.{20}f}  {wn_output[1]:.{20}f} \n")

                f.close()
            self.obs = old_obs
            self.LS = old_LS
           
        # Plotting the histograms of the Whittle likelihood distributions
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

                                            #FALS FROM AR(1)

    #=====================================================================================================================



    def fal_ar1(self,phi,sigma,n_bootstrap=10000,Plot=True,title="FALs based on AR(1)"):

       
        if ((type(phi) is np.ndarray) and (type(sigma) is np.ndarray)):
            try:
                    test = (len(phi) == len(sigma)==n_bootstrap)
                    if (not test):
                        raise ValueError
            except ValueError:
                    print("The length of distributions of phi and sigma should be equal to number of bootstraps")
                    return



        t = self.time[len(self.time)-1]-self.time[0] 


        rr = 0.5*(1/t)
        N_gridpoints = int((self.fnyq/rr)*3)
        fgrid = np.linspace(rr,self.fnyq,num=N_gridpoints,endpoint=True)

        if self.fmin!= "Default":   
            def find_index_closest_to_value(arr,target_value,tolerance):
                differences = np.abs(arr - target_value)
                min_index = np.argmin(differences)
                if differences[min_index] <= tolerance:
                    return min_index
                else:
                    raise ValueError('No grid point found near the specified fmin within the tolerence range')
            fgrid = fgrid[find_index_closest_to_value(fgrid,self.fmin,10**-4):]
        
        
        ls_real = LombScargle(self.time,self.obs,normalization="psd")
        LS_real = ls_real.power(fgrid)
        fal_white = ls_real.false_alarm_level([0.05, 0.01,0.001], method = 'bootstrap')


        n = len(self.time)             # No. of data points in timeseries

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
                delta_t = self.time[i]-self.time[i-1]
                new_obs[i] = new_obs[i-1]*np.exp(-delta_t/tau) + epsilon[i]

    # Normalizing 
            std_obs = np.std(new_obs)
            Obs = new_obs/std_obs

    # Generating the power spectrum for each timeseries and saving the powers across all frequency grid
            ls = LombScargle(self.time,Obs,normalization="psd")
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

                                            #FALS FROM POWER LAW

    #=====================================================================================================================

    def fal_pl(self,slope,intercept,n_bootstrap=10000,Plot=True,title="FALs based on power law"):


        if ((type(slope) is np.ndarray) and (type(intercept) is np.ndarray)):
            try:
                    test = (len(slope) == len(intercept)==n_bootstrap)
                    if (not test):
                        raise ValueError
            except ValueError:
                    print("The length of distributions of slope and intercept should be equal to number of bootstraps")
                    return

        t = self.time[len(self.time)-1]-self.time[0] 



        rr = 0.5*(1/t)
        N_gridpoints = int((self.fnyq/rr)*3)
        fgrid = np.linspace(rr,self.fnyq,num=N_gridpoints,endpoint=True)

        if self.fmin!= "Default":   
            def find_index_closest_to_value(arr,target_value,tolerance):
                differences = np.abs(arr - target_value)
                min_index = np.argmin(differences)
                if differences[min_index] <= tolerance:
                    return min_index
                else:
                    raise ValueError('No grid point found near the specified fmin within the tolerence range')
            fgrid = fgrid[find_index_closest_to_value(fgrid,self.fmin,10**-4):]


        ls_real = LombScargle(self.time,self.obs,normalization="psd")
        LS_real = ls_real.power(fgrid)
        fal_white = ls_real.false_alarm_level([0.05, 0.01,0.001], method = 'bootstrap')

        flog = np.log10(fgrid)

    # Linear function    
        def func(x,a,b):
                return a*x+b

        n = len(self.time)
        k = len(fgrid)
        spec = np.zeros((k,n_bootstrap))
        rand_int = np.random.randn(n_bootstrap)


        time = self.time-self.time[0]
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

