import torch
import numpy as np
import scipy.stats

def statError(pred, target):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ngrid, nt = pred.shape
        # Bias
        Bias = np.nanmean(pred - target, axis=1)
        # RMSE
        RMSE = np.sqrt(np.nanmean((pred - target) ** 2, axis=1))
        # ubRMSE
        predMean = np.tile(np.nanmean(pred, axis=1), (nt, 1)).transpose()
        targetMean = np.tile(np.nanmean(target, axis=1), (nt, 1)).transpose()
        predAnom = pred - predMean
        targetAnom = target - targetMean
        ubRMSE = np.sqrt(np.nanmean((predAnom - targetAnom) ** 2, axis=1))

        # # FDC metric
        # predFDC = calFDC(pred)
        # targetFDC = calFDC(target)
        # FDCRMSE = np.sqrt(np.nanmean((predFDC - targetFDC) ** 2, axis=1))

        # rho R2 NSE
        Corr = np.full(ngrid, np.nan)
        CorrSp = np.full(ngrid, np.nan)
        R2 = np.full(ngrid, np.nan)
        NSE = np.full(ngrid, np.nan)
        PBiaslow = np.full(ngrid, np.nan)
        PBiashigh = np.full(ngrid, np.nan)
        PBias = np.full(ngrid, np.nan)
        PBiasother = np.full(ngrid, np.nan)
        KGE = np.full(ngrid, np.nan)
        KGE12 = np.full(ngrid, np.nan)
        RMSElow = np.full(ngrid, np.nan)
        RMSEhigh = np.full(ngrid, np.nan)
        RMSEother = np.full(ngrid, np.nan)
        for k in range(0, ngrid):
            x = pred[k, :]
            y = target[k, :]
            ind = np.where(np.logical_and(~np.isnan(x), ~np.isnan(y)))[0]
            if ind.shape[0] > 0:
                xx = x[ind]
                yy = y[ind]
                # percent bias
                PBias[k] = np.sum(xx - yy) / np.sum(yy) * 100

                # FHV the peak flows bias 2%
                # FLV the low flows bias bottom 30%, log space
                pred_sort = np.sort(xx)
                target_sort = np.sort(yy)
                indexlow = round(0.3 * len(pred_sort))
                indexhigh = round(0.98 * len(pred_sort))
                lowpred = pred_sort[:indexlow]
                highpred = pred_sort[indexhigh:]
                otherpred = pred_sort[indexlow:indexhigh]
                lowtarget = target_sort[:indexlow]
                hightarget = target_sort[indexhigh:]
                othertarget = target_sort[indexlow:indexhigh]
                PBiaslow[k] = np.sum(lowpred - lowtarget) / np.sum(lowtarget) * 100
                PBiashigh[k] = np.sum(highpred - hightarget) / np.sum(hightarget) * 100
                PBiasother[k] = np.sum(otherpred - othertarget) / np.sum(othertarget) * 100
                RMSElow[k] = np.sqrt(np.nanmean((lowpred - lowtarget) ** 2))
                RMSEhigh[k] = np.sqrt(np.nanmean((highpred - hightarget) ** 2))
                RMSEother[k] = np.sqrt(np.nanmean((otherpred - othertarget) ** 2))

                if ind.shape[0] > 1:
                    # Theoretically at least two points for correlation
                    Corr[k] = scipy.stats.pearsonr(xx, yy)[0]
                    CorrSp[k] = scipy.stats.spearmanr(xx, yy)[0]
                    yymean = yy.mean()
                    yystd = np.std(yy)
                    xxmean = xx.mean()
                    xxstd = np.std(xx)
                    KGE[k] = 1 - np.sqrt((Corr[k] - 1) ** 2 + (xxstd / yystd - 1) ** 2 + (xxmean / yymean - 1) ** 2)
                    KGE12[k] = 1 - np.sqrt((Corr[k] - 1) ** 2 + ((xxstd * yymean) / (yystd * xxmean) - 1) ** 2 + (
                                xxmean / yymean - 1) ** 2)
                    SST = np.sum((yy - yymean) ** 2)
                    SSReg = np.sum((xx - yymean) ** 2)
                    SSRes = np.sum((yy - xx) ** 2)
                    R2[k] = 1 - SSRes / SST
                    NSE[k] = 1 - SSRes / SST

        outDict = dict(Bias=Bias, RMSE=RMSE, ubRMSE=ubRMSE, Corr=Corr, CorrSp=CorrSp, R2=R2, NSE=NSE,
                       FLV=PBiaslow, FHV=PBiashigh, PBias=PBias, PBiasother=PBiasother, KGE=KGE, KGE12=KGE12,
                       lowRMSE=RMSElow, highRMSE=RMSEhigh, midRMSE=RMSEother)  # fdcRMSE=FDCRMSE,
    return outDict

"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Hochreiter, S., and Nearing, G. S.: A note on leveraging synergy in multiple meteorological
datasets with deep learning for rainfall-runoff modeling, Hydrol. Earth Syst. Sci. Discuss.,
https://doi.org/10.5194/hess-2020-221, in review, 2020.

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""
import numpy as np
from scipy import stats, signal
from xarray.core.dataarray import DataArray


def _validate_inputs(obs: DataArray, sim: DataArray):
    if obs.shape != sim.shape:
        raise RuntimeError("Shapes of observations and simulations must match")

    if (len(obs.shape) > 1) and (obs.shape[1] > 1):
        raise RuntimeError(
            "Metrics only defined for time series (1d or 2d with second dimension 1)")


def _mask_valid(obs: DataArray, sim: DataArray, remove_neg=True) -> (DataArray, DataArray):

    # mask of invalid entries
    if remove_neg:
        idx = (obs >= 0) & (~obs.isnull())
    else:
        idx = ~obs.isnull()

    obs = obs[idx]
    sim = sim[idx]

    return obs, sim


def _get_fdc(da: DataArray) -> np.ndarray:
    return da.sortby(da, ascending=False).values


def nse(obs: DataArray, sim: DataArray, remove_neg=True) -> float:
    # verify inputs
    _validate_inputs(obs, sim)

    # get time series with only valid observations
    obs, sim = _mask_valid(obs, sim, remove_neg=remove_neg)

    denominator = ((obs - obs.mean()) ** 2).sum()
    numerator = ((sim - obs) ** 2).sum()

    value = 1 - numerator / denominator

    return float(value)


def mse(obs: DataArray, sim: DataArray, remove_neg=True) -> float:
    # verify inputs
    _validate_inputs(obs, sim)

    # get time series with only valid observations
    obs, sim = _mask_valid(obs, sim, remove_neg=remove_neg)

    return float(((sim - obs) ** 2).mean())


def rmse(obs: DataArray, sim: DataArray) -> float:
    return np.sqrt(mse(obs, sim))


def alpha_nse(obs: DataArray, sim: DataArray, remove_neg=True) -> float:
    # verify inputs
    _validate_inputs(obs, sim)

    # get time series with only valid observations
    obs, sim = _mask_valid(obs, sim, remove_neg=remove_neg)

    return float(sim.std() / obs.std())


def beta_nse(obs: DataArray, sim: DataArray, remove_neg=True) -> float:
    # verify inputs
    _validate_inputs(obs, sim)

    # get time series with only valid observations
    obs, sim = _mask_valid(obs, sim, remove_neg=remove_neg)

    return float((sim.mean() - obs.mean()) / obs.std())


def beta_kge(obs: DataArray, sim: DataArray, remove_neg=True) -> float:
    # verify inputs
    _validate_inputs(obs, sim)

    # get time series with only valid observations
    obs, sim = _mask_valid(obs, sim, remove_neg=remove_neg)

    return float(sim.mean() / obs.mean())


def kge(obs: DataArray, sim: DataArray, weights: list = [1, 1, 1], remove_neg=True) -> float:
    if len(weights) != 3:
        raise ValueError("Weights of the KGE must be a list of three values")

    # verify inputs
    _validate_inputs(obs, sim)

    # get time series with only valid observations
    obs, sim = _mask_valid(obs, sim, remove_neg=remove_neg)

    if len(obs.values) >= 2 and len(sim.values) >= 2:
        r, _ = stats.pearsonr(obs.values, sim.values)
    else:
        return np.nan

    # r, _ = stats.pearsonr(obs.values, sim.values)

    alpha = sim.std() / obs.std()
    beta = sim.mean() / obs.mean()

    value = (weights[0] * (r - 1) ** 2 + weights[1] * (alpha - 1) ** 2 + weights[2] * (beta - 1) ** 2)

    return 1 - np.sqrt(float(value))


def pearsonr(obs: DataArray, sim: DataArray, remove_neg=True) -> float:
    # verify inputs
    _validate_inputs(obs, sim)

    # get time series with only valid observations
    obs, sim = _mask_valid(obs, sim, remove_neg=remove_neg)

    # r, _ = stats.pearsonr(obs.values, sim.values)

    if len(obs.values) >= 2 and len(sim.values) >= 2:
        r, _ = stats.pearsonr(obs.values, sim.values)
    else:
        return np.nan

    return float(r)


def fdc_fms(obs: DataArray, sim: DataArray, lower: float = 0.2, upper: float = 0.7, remove_neg=True) -> float:
    """Slope of the middle section of the flow duration curve.

    Reference:
    Yilmaz, K. K., Gupta, H. V., and Wagener, T. ( 2008), A process‐based diagnostic approach to model evaluation:
    Application to the NWS distributed hydrologic model, Water Resour. Res., 44, W09417, doi:10.1029/2007WR006716.
    """
    # verify inputs
    _validate_inputs(obs, sim)

    # get time series with only valid observations
    obs, sim = _mask_valid(obs, sim, remove_neg=remove_neg)

    if any([(x <= 0) or (x >= 1) for x in [upper, lower]]):
        raise ValueError("upper and lower have to be in range ]0,1[")

    if lower >= upper:
        raise ValueError("The lower threshold has to be smaller than the upper.")

    # get arrays of sorted (descending) discharges
    obs = _get_fdc(obs)
    sim = _get_fdc(sim)

    # for numerical reasons change 0s to 1e-6. Simulations can still contain negatives, so also reset those.
    sim[sim <= 0] = 1e-6
    obs[obs == 0] = 1e-6

    # calculate fms part by part
    qsm_lower = np.log(sim[np.round(lower * len(sim)).astype(int)])
    qsm_upper = np.log(sim[np.round(upper * len(sim)).astype(int)])
    qom_lower = np.log(obs[np.round(lower * len(obs)).astype(int)])
    qom_upper = np.log(obs[np.round(upper * len(obs)).astype(int)])

    fms = ((qsm_lower - qsm_upper) - (qom_lower - qom_upper)) / (qom_lower - qom_upper + 1e-6)

    return fms * 100


def fdc_fhv(obs: DataArray, sim: DataArray, h: float = 0.02, remove_neg=True) -> float:
    """Peak flow bias derived from the flow duration curve.

    Reference:
    Yilmaz, K. K., Gupta, H. V., and Wagener, T. ( 2008), A process‐based diagnostic approach to model evaluation:
    Application to the NWS distributed hydrologic model, Water Resour. Res., 44, W09417, doi:10.1029/2007WR006716.
    """
    # verify inputs
    _validate_inputs(obs, sim)

    # get time series with only valid observations
    obs, sim = _mask_valid(obs, sim, remove_neg=remove_neg)

    if (h <= 0) or (h >= 1):
        raise ValueError(
            "h has to be in range ]0,1[. Consider small values, e.g. 0.02 for 2% peak flows")

    # get arrays of sorted (descending) discharges
    obs = _get_fdc(obs)
    sim = _get_fdc(sim)

    # subset data to only top h flow values
    obs = obs[:np.round(h * len(obs)).astype(int)]
    sim = sim[:np.round(h * len(sim)).astype(int)]

    # disable warnings for division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        fhv = np.sum(sim - obs) / np.sum(obs)

    return fhv * 100


def fdc_flv(obs: DataArray, sim: DataArray, l: float = 0.3, remove_neg=True) -> float:
    """Low flow bias derived from the flow duration curve.

    Reference:
    Yilmaz, K. K., Gupta, H. V., and Wagener, T. ( 2008), A process‐based diagnostic approach to model evaluation:
    Application to the NWS distributed hydrologic model, Water Resour. Res., 44, W09417, doi:10.1029/2007WR006716.
    """
    # verify inputs
    _validate_inputs(obs, sim)

    # get time series with only valid observations
    obs, sim = _mask_valid(obs, sim, remove_neg=remove_neg)

    if (l <= 0) or (l >= 1):
        raise ValueError(
            "l has to be in range ]0,1[. Consider small values, e.g. 0.3 for 30% low flows")

    # get arrays of sorted (descending) discharges
    obs = _get_fdc(obs)
    sim = _get_fdc(sim)

    # for numerical reasons change 0s to 1e-6. Simulations can still contain negatives, so also reset those.
    sim[sim <= 0] = 1e-6
    obs[obs == 0] = 1e-6

    obs = obs[-np.round(l * len(obs)).astype(int):]
    sim = sim[-np.round(l * len(sim)).astype(int):]

    # transform values to log scale
    obs = np.log(obs)
    sim = np.log(sim)

    # calculate flv part by part
    qsl = np.sum(sim - sim.min())
    qol = np.sum(obs - obs.min())

    flv = -1 * (qsl - qol) / (qol + 1e-6)

    return flv * 100


def mean_peak_timing(obs: DataArray, sim: DataArray, window: int = 3,
                     resolution: str = 'D', remove_neg=True) -> float:
    """Absolute peak timing error

    My own metrics: Uses scipy to find peaks in the observed flows and then computes the absolute time difference
    between the observed flow and the day in which the peak is found in the simulation time series.

    """
    # verify inputs
    _validate_inputs(obs, sim)

    # get time series with only valid observations
    obs, sim = _mask_valid(obs, sim, remove_neg=remove_neg)

    # get indices of peaks and their corresponding height
    peaks, properties = signal.find_peaks(obs.values, distance=100, height=(None, None))

    # evaluate timing
    timing_errors = []
    for idx, height in zip(peaks, properties['peak_heights']):

        # skip peaks at the start and end of the sequence
        if (idx - window < 0) or (idx + window > len(obs)):
            continue

        # check if the value at idx is a peak (both neighbors must be smaller)
        if (sim[idx] > sim[idx - 1]) and (sim[idx] > sim[idx + 1]):
            peak_sim = sim[idx]
        else:
            # define peak around idx as the max value inside of the window
            values = sim[idx - window:idx + window + 1]
            peak_sim = values[values.argmax()]

        # get xarray object of qobs peak, for getting the date and calculating the datetime offset
        peak_obs = obs[idx]

        # calculate the time difference between the peaks
        delta = peak_obs.coords['date'] - peak_sim.coords['date']

        timing_error = np.abs(
            delta.values.astype(f'timedelta64[{resolution}]') / np.timedelta64(1, resolution))

        timing_errors.append(timing_error)

    return np.sum(timing_errors) / len(peaks)


def calculate_all_metrics(obs: DataArray, sim: DataArray, remove_neg=True) -> dict:
    """Calculate all metrics with default values."""
    import xarray as xr
    if type(obs) != xr.DataArray:
        obs = xr.DataArray(obs)
    if type(sim) != xr.DataArray:
        sim = xr.DataArray(sim)

    results = {
        "NSE": nse(obs, sim, remove_neg=remove_neg),
        "MSE": mse(obs, sim, remove_neg=remove_neg),
        "RMSE": rmse(obs, sim),
        "KGE": kge(obs, sim, remove_neg=remove_neg),
        "Alpha-NSE": alpha_nse(obs, sim, remove_neg=remove_neg),
        "Beta-NSE": beta_nse(obs, sim, remove_neg=remove_neg),
        "Corr": pearsonr(obs, sim, remove_neg=remove_neg),
        "FHV": fdc_fhv(obs, sim, remove_neg=remove_neg),
        # "FMS": fdc_fms(obs, sim, remove_neg=remove_neg),
        # "FLV": fdc_flv(obs, sim, remove_neg=remove_neg),
        # "Peak-Timing": mean_peak_timing(obs, sim)
    }

    return results


def cal_stations_metrics(obs_all, sim_all, metrics_list=None, remove_neg=True):
    """
    NSE_list, KGE_list, RMSE = [], [], []
    for idx_basin, basin in enumerate(basins):
        qobs = obs[idx_basin]
        qsim = pred[idx_basin]

        values = calculate_all_metrics(qobs, qsim)

        NSE_list.append(values["NSE"])
        KGE_list.append(values["KGE"])
        RMSE.append(values["RMSE"])

    NSE = np.array(NSE_list)
    KGE = np.array(KGE_list)
    RMSE = np.array(RMSE)
    """
    # obs_all: [basin, time]
    if metrics_list is None:
        metrics_list = ['NSE', 'MSE', 'RMSE', 'KGE', 'Alpha-NSE', 'Beta-NSE', 'Pearson r', 'FHV', 'FMS', 'FLV']

    num_sites = obs_all.shape[0]

    metrics_dict = {key: [] for key in metrics_list}
    for idx_basin in range(num_sites):
        qobs = obs_all[idx_basin]
        qsim = sim_all[idx_basin]

        # calculate nan ratio
        nan_ratio_obs = np.sum(np.isnan(qobs)) / len(qobs)
        nan_ratio_sim = np.sum(np.isnan(qsim)) / len(qsim)
        # print(f"{idx_basin}: nan ratio: obs {nan_ratio_obs}, sim {nan_ratio_sim}")
        if (nan_ratio_obs == 1) | (nan_ratio_sim == 1):
            metrics_dict = {key: metrics_dict[key] + [np.nan] for key in metrics_list}
            print(f"{idx_basin}: nan ratio is 1, skip")
            continue
            # return metrics_dict

        values = calculate_all_metrics(qobs, qsim, remove_neg=remove_neg)
        metrics_dict = {key: metrics_dict[key] + [values[key]] for key in metrics_list}

    metrics_dict = {key: np.array(metrics_dict[key]) for key in metrics_list}

    return metrics_dict