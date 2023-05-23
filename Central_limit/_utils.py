import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

class population_gens:
    def __init__(self, population_size):
        self.data = None
        self.population_size = population_size
    def gen_population_square(self): # 平均分佈
        population_square = np.ones(self.population_size)
        for i in range(len(population_square)):
            tmp = (i/len(population_square))
            population_square[i] = population_square[i]*tmp

        out = {'data': population_square, 'name': "population_square"}
        self.data = out
        return out
    
    def gen_population_triangle(self): # 三角分佈
        H = 0
        tmp = 0
        H = int(np.floor(np.sqrt(self.population_size)))
        total_num = H**2
        population_tri = []
        for i in range(1,H*2+1):
            if i <= H:
                for _ in range(i):
                    population_tri.append(i/H/2)
            else:
                for _ in range(H, i-H, -1):
                    population_tri.append(i/H/2)

        population_tri = np.concatenate((population_tri, np.zeros(self.population_size-total_num)))
        out = {'data': population_tri, 'name': "population_tri"}
        self.data = out
        return out
    
    def gen_population_normal(self): # 常態分佈
        population_norm = np.random.normal(0, 1, self.population_size)
        out = {'data': population_norm, 'name': "population_norm"}
        self.data = out
        return out

    def gen_population_rand(self): # 隨機分佈
        population_rand = np.random.rand(self.population_size)
        out = {'data': population_rand, 'name': "population_rand"}
        self.data = out
        return out

    def gen_my_population(self, data, name = "my_population"): # 自訂分佈
        if len(data) != self.population_size:
            raise Exception(f"Data size ({len(data)}) not match self.population_size ({self.population_size})")
        out = {'data': data, 'name': name}
        self.data = out
        return out

    def plot(self, data = None, name = "", plot_mu_sig = False):
        if data is None:
            data = self.data["data"]
            name = self.data["name"]
        
        mu = np.mean(data)
        sigma = np.std(data)
        plt.suptitle('Population type: ' + str(name), fontsize=16, y=1.05)
        plt.subplot(1,2,1)
        sns.histplot(data,bins=int(50),kde = False)
        plt.title('Histogram of Population mean')
        plt.xlabel('Population mean')
        plt.ylabel('Count')
        if plot_mu_sig:
            tmp = str("{:.2e}".format(mu))
            plt.axvline(x=mu, color='orange', label = f"mean = {tmp}")
            plt.axvline(x=mu+sigma, color='g', linestyle='--', label = "mean+std")
            plt.axvline(x=mu-sigma, color='g', linestyle='--', label = "mean-std")
            plt.legend()
        print("Data mean: " + str("{:.2e}".format(mu)))
        print("Data std: " + str("{:.2e}".format(sigma)))
        plt.subplot(1,2,2)
        sns.kdeplot(data)
        plt.title('Density of Population mean')
        plt.xlabel('Population mean')
        plt.ylabel('Density')
        if plot_mu_sig:
            tmp = str("{:.2e}".format(mu))
            plt.axvline(x=mu, color='orange', label = f"mean = {tmp}")
            plt.axvline(x=mu+sigma, color='g', linestyle='--', label = "mean+std")
            plt.axvline(x=mu-sigma, color='g', linestyle='--', label = "mean-std")
            plt.legend()
        plt.subplots_adjust(bottom=0.1, right=2, top=0.9)
        plt.show()

    def plot_hist_multi(self, sample_size, number_of_samples, populations, sampling, cache = None):
        pad = 10 # in points
        fig, ax = plt.subplots(len(sample_size)+1,len(populations),figsize=(5, 12))
        for axn, col in zip(ax[0], populations):
            axn.annotate(col["name"], xy=(0.5, 1), xytext=(0, pad),
                        xycoords='axes fraction', textcoords='offset points',
                        fontsize='large', ha='center', va='baseline')
        sample_size_annotate = []
        for i in range(len(sample_size)+1):
            if i == 0:
                sample_size_annotate.append("Population")
            else:
                sample_size_annotate.append("Sample size\nN = "+str(sample_size[i-1]))

        for axn, row in zip(ax[:,0], sample_size_annotate):
            axn.annotate(row, xy=(0, 0.5), xytext=(-axn.yaxis.labelpad - pad, 0),
                        xycoords=axn.yaxis.label, textcoords='offset points',
                        size='large', ha='right', va='center')


        sample_mean_cache = cache if cache is not None else []
        for j in range(0,len(populations)):
            plt.subplot(len(sample_size)+1,len(populations),j+1)
            if j == 0:
                plt.ylabel('Count')
            else:
                plt.ylabel(' ')
            
            sns.histplot(populations[j]["data"],bins=int(50),kde = True)
            plt.xlim(populations[j]["data"].min(),populations[j]["data"].max())
            plt.subplots_adjust(bottom=0.1, right=2, top=0.9)
            for i in range(0,len(sample_size)):
                if cache is None:
                    sample_means = sampling(populations[j]["data"], sample_size[i], number_of_samples)
                    sample_mean_cache.append(sample_means)
                else:
                    sample_means = sample_mean_cache[i*len(populations)+j]
                plt.subplot(len(sample_size)+1,len(populations),len(populations)*(i+1)+j+1)
                if j == 0:
                    plt.ylabel('Count')
                else:
                    plt.ylabel(' ')

                if i == len(sample_size)-1:
                    plt.xlabel('Sample mean')
                sns.histplot(sample_means,bins=int(50),kde = True)
                plt.xlim(populations[j]["data"].min(),populations[j]["data"].max())
                plt.subplots_adjust(bottom=0.1, right=2.5, top=0.9)

        return sample_mean_cache

    def plot_kde_multi(self, sample_size, number_of_samples, populations, sampling, cache = None):
        pad = 10 # in points
        fig, ax = plt.subplots(len(sample_size)+1,len(populations),figsize=(5, 12))
        for axn, col in zip(ax[0], populations):
            axn.annotate(col["name"], xy=(0.5, 1), xytext=(0, pad),
                        xycoords='axes fraction', textcoords='offset points',
                        fontsize='large', ha='center', va='baseline')
        sample_size_annotate = []
        for i in range(len(sample_size)+1):
            if i == 0:
                sample_size_annotate.append("Population")
            else:
                sample_size_annotate.append("Sample size\nN = "+str(sample_size[i-1]))

        for axn, row in zip(ax[:,0], sample_size_annotate):
            axn.annotate(row, xy=(0, 0.5), xytext=(-axn.yaxis.labelpad - pad, 0),
                        xycoords=axn.yaxis.label, textcoords='offset points',
                        size='large', ha='right', va='center')

        sample_mean_cache = cache if cache is not None else []
        for j in range(0,len(populations)):
            plt.subplot(len(sample_size)+1,len(populations),j+1)
            if j == 0:
                plt.ylabel('Density')
            else:
                plt.ylabel(' ')
            sns.kdeplot(populations[j]["data"])
            plt.xlim(populations[j]["data"].min(),populations[j]["data"].max())
            plt.subplots_adjust(bottom=0.1, right=2, top=0.9)
            for i in range(0,len(sample_size)):
                if cache is None:
                    sample_means = sampling(populations[j]["data"], sample_size[i], number_of_samples)
                    sample_mean_cache.append(sample_means)
                else:
                    sample_means = sample_mean_cache[i*len(populations)+j]
                plt.subplot(len(sample_size)+1,len(populations),len(populations)*(i+1)+j+1)
                if j == 0:
                    plt.ylabel('Density')
                else:
                    plt.ylabel(' ')
                if i == len(sample_size)-1:
                    plt.xlabel('Sample mean')
                sns.kdeplot(sample_means)
                plt.xlim(populations[j]["data"].min(),populations[j]["data"].max())
                plt.subplots_adjust(bottom=0.1, right=2.5, top=0.9)

        return sample_mean_cache