'''
Isaac Severinsen 23/05/2022

COVERT (Classless oversampling technique) is designed to reblance a dataset that does not include classes.
This was designed for balancing industrial steady state data so that rarer (startup and shutdown) conditions
are represented to the same degree as those gathered during normal operation.

Inputs:
- Dataframe of raw data
    -  Contains columns of all variables and associated columns of uncertainties,
    -  Values are suffixed with '_mean', uncertainties with '_std'
- input_list, list of variable names that define the input parameters only
- Various Optional Parameters
    -  verbose, bool, verbosity boolean, default is True
    -  rebalance_ratio, float determining how many synthetic samples to generate reblance_ratio = 1 means the number of synthetic samples generated is equal to the number of raw datapoints supplied.

'''

from scipy.stats import gaussian_kde, chisquare, kstest
from scipy.linalg import cholesky
from scipy.spatial.distance import mahalanobis, pdist, squareform
import numpy as np
import pandas as pd
from numba import jit
import time
import matplotlib.pyplot as plt

def pandas_wrapper(df, input_list, verbose, rebalance_ratio = 0):
    # just a simple wrapper to utilise pandas formatted data

    df, scaler_data = scale_data(df) # Scale data, scaler data is required to unscale data

    input_list_mean = [i for i in df.columns if i[:-5] in input_list]
    input_list_std = [i for i in df.columns if i[:-4] in input_list]

    mean_list = [i for i in df.columns if i[-5:]  == '_mean']
    std_list = [i for i in df.columns if i[-4:] == '_std']

    # inputs_mean = df[input_list_mean].to_numpy()
    data_mean = df[mean_list].to_numpy()
    data_std = df[std_list].to_numpy()

    input_inx = [df.columns.get_loc(c) for c in df.columns if c[:-5] in input_list]

    data_out, target, metrics = COVERT(input_inx, data_mean, data_std, rebalance_ratio, verbose=verbose)

    df_out = pd.DataFrame(data = data_out, columns = list(df.columns) + ['anchor', 'i_index', 'synthetic', 'synth_anchor', 'synth_anchor_count', 'rarity'])
    df_goal = pd.DataFrame(data = target, columns = input_list_mean)
    # convert outputs to dataframe again

    df_out = scale_data(df_out, scaler_data) # unscale data
    df_goal = scale_data(df_goal, scaler_data) # unscale data

    return df_out, df_goal, metrics

def COVERT(input_inx, data, data_std, rebalance_ratio = 0 , verbose = False):

    original_len = data.shape[0]
    over = True

    metrics = {'IAE':[],'CHI2':[],'KS':[], 'N':[]}

    rarity_arr, kernel_raw, rarity_dict = compute_rarity(data[:,input_inx], np.array([]), original_len)

    if rebalance_ratio == 0:
        target = target_distribution(data[:,input_inx], kernel_raw, verbose)
        IAE,CHI2,KS, rebalance_ratio = performance_metrics(data[:,input_inx], target)
    else:
        target = target_distribution(data[:,input_inx], kernel_raw, verbose)
        IAE,CHI2,KS, _ = performance_metrics(data[:,input_inx], target)

    anchor = synthetic = synth_anchor = synth_anchor_count = np.zeros(original_len)
    synthetic = synthetic.astype('bool')

    stop = False
    N = 0        # counter for synthetic samples generated.
    N_old = 0
    rarity_iter = 0
    N_max = int(original_len * rebalance_ratio)
    recalc_rate = int(0.1*original_len) # this just alters the speed of the calculation - unlikely to need to be changed.
    num_neighbours = int(0.1*original_len) # this is based off the original data as the neighbours can only be real data.
    batch_size = int(np.ceil(recalc_rate/100)) # 100 batches between recalulate -remebering that synthetic repetition could decrease this significantly.

    print('recalc_rate', recalc_rate)
    print('batch_size', batch_size)
    print('num_neighbours', num_neighbours)

    inx_near_arr, inx_far_arr, cov = find_neighbours(data[:,input_inx], num_neighbours)

    if N_max == 0:
        stop = True

    i_index = np.arange(original_len)

    while not stop:
        
        anchor_list = choose_anchors(data[:,input_inx], anchor, i_index, synthetic, synth_anchor, synth_anchor_count, rarity_arr, batch_size, over)
        N_old = N

        if over:
            data, data_std, anchor_list, synthetic, anchor, rarity_arr, synth_anchor, synth_anchor_count, N = oversample(data[:,input_inx], data,
                  data_std, anchor_list, synthetic, anchor, rarity_arr, inx_near_arr, inx_far_arr, synth_anchor, synth_anchor_count, N, num_neighbours, rebalance_ratio )
        else:
            df, N, stop = undersample(df, anchor_list, N, N_max, stop)

        IAE,CHI2,KS, _ = performance_metrics(data[:,input_inx], target)

        for metric,code in zip([IAE,CHI2,KS],['IAE','CHI2','KS']):
            metrics[code].append(metric)
        metrics['N'].append(N)

        if N//recalc_rate > rarity_iter or N == 0:
            print('recalculating at:',N)
            N_new = original_len + N
            rarity_iter = N//recalc_rate
        else:
            N_new = N - N_old

        i_index = np.arange(data.shape[0]) # reindex the dataset with a custom column
        rarity_arr, kernel, rarity_dict = compute_rarity(data[:,input_inx], rarity_arr, N_new, rarity_dict)

        # this is here because the 
        if N >= N_max:# stop if number generated is higher that maximum.
            stop = True
            break 

    
    i_index = np.arange(data.shape[0])
    if over:
        if N >= N_max:
            final_len = original_len+N_max
        else:
            final_len = original_len + N
    else:
        final_len = data.shape[0]

    concat = np.concatenate((anchor[:, np.newaxis], i_index[:, np.newaxis], synthetic[:, np.newaxis], 
                            synth_anchor[:, np.newaxis], synth_anchor_count[:, np.newaxis], rarity_arr[:, np.newaxis]), axis = 1) 
    
    data_out = np.concatenate((data[:final_len], data_std[:final_len],concat[:final_len]),axis = 1)

    return data_out, target, metrics

def scale_data(df, *scaler_data):
    '''
    Designed to scale and unscale data using a range scaling method
    The optional argument is for unscaling as the original values are required.
    '''
    
    if scaler_data: # unscale
        scaler_save_divide = scaler_data[0][0]
        scaler_save_subtract = scaler_data[0][1]
        for var in df.columns:
            if '_mean' in var:
                df[var] = df[var] * scaler_save_divide[var] + scaler_save_subtract[var]
            elif '_std' in var:
                df[var] = df[var] * scaler_save_divide[var[:-4] + '_mean']
        return df

    else:  # scale
        mean_list = [c for c in df.columns if 'mean' in c]
        scaler_save_subtract = dict(zip(mean_list, df[mean_list].min().to_numpy()))
        scaler_save_divide = dict(zip(mean_list, (df[mean_list].max() - df[mean_list].min()).to_numpy()))

        for var in df.columns:
            if '_mean' in var:
                df[var] = (df[var] - scaler_save_subtract[var]) / scaler_save_divide[var]
            elif '_std' in var:
                df[var] = (df[var]) / scaler_save_divide[var[:-4] + '_mean']
        return df, (scaler_save_divide, scaler_save_subtract)

def compute_rarity(inputs, rarity_arr, N_new, rarity_dict = {}):

    kernel = gaussian_kde(inputs.T, bw_method = 'scott')
    # kernel.set_bandwidth(kernel.factor/1) incase we wanted to change the bandwidth of the kde
    kde = kernel.evaluate(inputs[-N_new:,:].T)

    if inputs.shape[0] == N_new:
        maxx = kde.max()
        minn = kde.min()
        rarity_dict = {'max':maxx, 'min':minn}
        rarity_arr = 1-((kde - rarity_dict['min'])/(rarity_dict['max'] - rarity_dict['min']))
        
    else:
        kde_new = 1-((kde - rarity_dict['min'])/(rarity_dict['max'] - rarity_dict['min']))
        rarity_arr = np.append(rarity_arr,kde_new)

    return rarity_arr, kernel, rarity_dict

scaler = lambda x: (x - x.min())/np.ptp(x)

def find_neighbours(inputs, num_neighbours):

    N = len(inputs)
    cov = np.cov(inputs, rowvar=False)
    # dist_list = squareform(pdist(inputs, 'mahalanobis', VI = cov))
    dist_list = squareform(pdist(inputs, 'euclidean'))

    inx = np.zeros((N, num_neighbours))
    inx[:N,:] = np.argpartition(dist_list,num_neighbours)[:,:num_neighbours]
    inx = inx.astype('int32')

    inx_near_arr = inx.tolist()
    inx_near_arr = [np.array(x) for x in inx]
    inx_near_arr = [x[x>0] for x in inx]

    inx_far_arr = sort_inx(inx_near_arr, inputs, dist_list)

    return inx_near_arr, inx_far_arr, cov

@jit(nopython=True)
def arg_sortt(rare_list):
    return np.argsort(rare_list)

def sort_inx(inx_near_arr, inputs, dist_list):
    for i in range(len(inx_near_arr)):
        rare_list = squareform(pdist(inputs[inx_near_arr[i],:], 'euclidean'))
        # rare_list = squareform(pdist(inputs[inx_near_arr[i],:], 'mahalanobis'))
        rare_list_1D = 1-scaler(np.mean(rare_list, axis = 0 )) # metric describing how rare each neighbour is within a subset of all neighbours
        reinx = arg_sortt(rare_list_1D + 1.0 * scaler(dist_list[inx_near_arr[i],i])) # we combine the metrics of rare neighbours and nearby neighbours
        inx_near_arr[i] = list(np.array(inx_near_arr[i])[reinx])
    return inx_near_arr

def target_distribution(inputs, kde_raw, verbose = False):

    kde_raw.set_bandwidth(kde_raw.factor/0.5) # helps smooth things a bit
    dim = inputs.shape[1]
    rarity_raw = kde_raw.evaluate(inputs.T)
    pct = np.percentile(rarity_raw, 10) # might be worth changing this

    NN = 100000 # should keep excecution time under control
    N = int(NN**(1/dim)) # in high dimensions this might be very low...
    min_list = np.min(inputs, axis = 0)
    max_list = np.max(inputs, axis = 0)
    data_list = [np.linspace(minn,maxx,N) for minn, maxx in zip(min_list, max_list)]
    mesh_data = np.meshgrid(*data_list)
    XX = np.array(mesh_data).reshape(dim, N**dim)

    rarity = kde_raw.evaluate(XX)
    target = XX[:,rarity > pct]
    loss_ratio = len(target)/len(rarity)

    if verbose:
        print('loss ratio',loss_ratio * 100,'%')

    return target.T

def performance_metrics(inputs, target):
    dim = inputs.shape[1] # number of variables 
    Ne = 1001 # this is a relatively low number but any higher and it gets very slow to calculate
    edges = [np.linspace(-1,1,Ne) for i in range(dim)]
    N = Ne-1

    H1, edges = np.histogramdd(target,bins=edges,normed = True)
    H1 = H1.reshape(N**dim)
    H2, edges = np.histogramdd(inputs,bins=edges,normed = True)
    H2 = H2.reshape(N**dim)

    inx = ~(H1==0).astype('bool')
    H1 = H1[inx]
    H2 = H2[inx]

    CDF1 = [0]
    CDF2 = [0]
    for i in H1:
        CDF1.append(CDF1[-1]+i)
    for j in H2:
        CDF2.append(CDF2[-1]+j)

    IAE = np.trapz(np.abs(np.array(CDF2)-np.array(CDF1)))
    CHI2 = np.sum((H2-H1)**2/H1)
    KS = kstest(CDF2, CDF1)[0]

    rebalance_ratio = np.sum(CDF1) / np.sum(CDF2)

    return IAE,CHI2,KS, rebalance_ratio

def ewm_fun(arr_in): # self containted exponential weeighted moving average to give a half life to data
    n = arr_in.shape[0]
    ewma = np.empty(n)
    alpha = 1 - np.exp(np.log(0.5)/0.5)
    w = 1
    ewma_old = arr_in[0]
    ewma[0] = ewma_old
    for i in range(1, n):
        w += (1-alpha)**i
        ewma_old = ewma_old*(1-alpha) + arr_in[i]
        ewma[i] = ewma_old / w
    return ewma

def choose_anchors(inputs, anchor, i_index, synthetic, synth_anchor, synth_anchor_count, rarity_arr, batch_size, over):
    '''
    The goal of this function is to, as the name suggests, choose anchors.
    Anchors are the basis points for synthetic sample generation.
    To achieve the target distribution the rarest points in the distribution are desirable anchors.
    Calculating rarity is expensive and to partially make up for this a rarity estimation is applied to previously selected anchors.
    This assigns new rarity values to previously selected anchors based on the rarity of the synthetic samples generated by this anchor.
    Subsequently anchors are chosen by simply selecting the rarest real points.
    '''

    # recalculate rarity for previously selected anchors.
    for i in i_index[anchor>0]: # looping through all previsouly selected anchors
        # selecting the rarity of synthietic points generated by the specified anchor
        y_raw = rarity_arr[synth_anchor == i] # selecting only those synthetic entries created by the anchor
        x_raw = synth_anchor_count[synth_anchor == i]

        if x_raw.shape[0] < 1:
            pass
        else:
            y = np.array([np.mean(y_raw[x_raw == j]) for j in set(x_raw)])
            # goal of this part is to take the average (ewm) of rarities as related to its number in synthetic anchor count.
            ewm = ewm_fun(y) # use self containted exponentially weighted moving average funciton.
            rarity_arr[i] = ewm[-1] # select the final value as the new points' rarity.
            
    rarity_arr = scaler(rarity_arr) # rescale as synthetic points may have rarity above 1 and thus ewm of these may results in rarity >1

    if over: # oversampling
        max_pool = np.argpartition(rarity_arr[~synthetic], -batch_size)[-batch_size:] # returns the indicies of rarest points (original data only)
        anchor_list = []
        for i in max_pool: # accounts for mismatch in index from sliced dataframe - somewhat deprecated but useful for safety.
            anchor_list.append(i_index[~synthetic][i])
        return anchor_list
        # could remove anchors that are near other anchors to avoid clustering in each batch before a rarity update

    else: # undersampling - can be real or synthetic
        if batch_size > input.shape[0]: # batch size too big, reduce it here
            batch_size = int(0.95*input.shape[0]-1)

        min_pool = np.argpartition(rarity_arr, batch_size)[:batch_size] # returns the indicies of the most common points
        under_list = np.random.choice(min_pool, int(0.8*len(min_pool)), replace=False) # add some randomness here - if undersampling other randomness will be needed to avoid blank areas.
        return under_list

def oversample(inputs, data, data_std, anchor_list, synthetic, anchor, rarity_arr, inx_near_arr, inx_far_arr, synth_anchor, synth_anchor_count, N_over, num_neighbours, rebalance_ratio ):
    '''
    This function, given a list of anchors and a list of nearest neighbours of each anchor will generate synthetic data.

    It first loops through all the selected anchor points, based on how many times this anchor has been selected it will adjust
    the number of neighbour hood points to allow and the degree to which they are combined.

    The covariance of all the local points is determined, if this fails the global covariance is used.
    It typically fails for a small numbers of neighbours.

    Then for each synthetic point neighbours are chosen randomly from the pool and
    the means and standard deviations of each variable are averaged based on the weighting established.

    Points are radnomly distributed around a mean zero point based on the anchor/neighbours' uncertainty in each direction.
    The synthetic point is then skewed used the covariance with a cholesky decomposition, then rescaled to each variables' units.
    Finally the mean (of the anchor/neighbours) of each variable is added and the variable inclued in the dataframe including a boolean to define this data as synthetic.
    Lastly the anchor counter is incrimented for the anchor point to de-incentivise repeated anchor points and more diverse sythetically generated data.
    '''

    dim = inputs.shape[1]
    synthetic_repeats_max = np.ceil(5**dim)
    acc = 1

    data_real = data[~synthetic]

    for i in anchor_list: # multithread from here? batch_size = threads?

        anchor[i] += 1
        N = anchor[i]
        rarity = rarity_arr[i]

        rarity = max(0,min(1,rarity))
        synthetic_repeats = int(max(min(synthetic_repeats_max,N*rarity * rebalance_ratio),1))
        repeat_weight = min(max((-(1/(50*rebalance_ratio))*(N-3)+1)*rarity,0.05),0.98)

        # a little convoluted but tshe 30 represents the number of iterations expected between the anchor and the neighbouring points
        # 3 is the number of iterations where only the anchor is used ot get started.

        cov = np.cov(data_real, rowvar=False) # only the shape of original datapoints
        H = cholesky(cov, lower=False) # cholesky decomposition to determine H matrix

        for j in range(synthetic_repeats): # iterating of number of synthetic repeats to do this means the same anchor is used but different directional anchors are used.
            frac = min(0.99,np.random.beta(1, 3)) # fraction of way through number of neighbours
            inx_dir = inx_far_arr[i][int(np.ceil(frac*num_neighbours))] # choses direction by selection directional anchor from sorted list
            NN2 = max(1,int(frac * 10)) # chooses number of directional anchor neighbours based on rarity of chosen directional anchor
            inx_near_rand = inx_near_arr[inx_dir][:NN2] # create list of neighbouring points' indicies
            inx_near_rand.append(inx_dir) # add the actual neighbour too.

            # Mix together anchor and local points as anchor points are repeated
            base_local = np.mean(data[inx_near_rand],axis = 0)
            base_anchor = data[i]
            base = base_anchor * repeat_weight + base_local * (1-repeat_weight)

            std_local = np.mean(data_std[inx_near_rand],axis = 0)
            std_anchor = data_std[i]
            std = std_anchor * repeat_weight + std_local * (1-repeat_weight)

            # if reasonable estimates of uncertainty are available
            rand1 = np.random.normal(0.0,acc*std)
            rand2 = np.matmul(rand1,H)/np.sqrt(np.diagonal(cov)) # skew and rescale the results
            final = base + rand2 # with cholesky skew
            # final = base + rand1 # without cholesky skew

            if (final < 0).any() or (final > 1).any(): # these values are inappropriate
                # print('drop')
                pass
            else:

                synthetic = np.append(synthetic, True)
                anchor = np.append(anchor, 0)
                synth_anchor = np.append(synth_anchor, i)
                synth_anchor_count = np.append(synth_anchor_count, N)

                data = np.concatenate((data, final[np.newaxis,:]), axis = 0) 
                data_std = np.concatenate((data_std, std[np.newaxis,:]), axis = 0) 

                N_over = N_over + 1
            # iter += 1

    return data, data_std, anchor_list, synthetic, anchor, rarity_arr, synth_anchor, synth_anchor_count, N_over

def undersample(data, under_list, N_under, N_max, stop):

    if len(under_list) + N_under > N_max:
        val = N_max - N_under -1
        N_under = N_under + val+1
        data = np.delete(data, under_list[:val], axis=0)
        stop = True
    else:
        N_under = N_under + len(under_list)
        data = np.delete(data, under_list, axis=0)
    return data, N_under, stop
