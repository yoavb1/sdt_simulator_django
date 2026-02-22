import json
import numpy as np
from django.shortcuts import render
from django.http import JsonResponse
import math
from core.architecture_realization import *


def index(request):
    """Renders the main dashboard."""
    return render(request, 'index.html')

def norm_ppf(p):
    """
    High-precision Inverse Normal CDF for Python < 3.12.
    Uses the relationship: ppf(p) = sqrt(2) * erfinv(2p - 1)
    Approximated via the high-accuracy Acklam's method.
    """
    if p <= 0 or p >= 1: return float('nan')

    # Coefficients for low/middle/high regions
    a = [-3.969683028665376e+01,  2.209460984245205e+02, -2.759285104469687e+02,
          1.383577518672690e+02, -3.066479806614716e+01,  2.506628277459239e+00]
    b = [-5.447609879442702e+01,  1.615858368580409e+02, -1.556989798598866e+02,
          6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00]
    d = [ 7.784695709041462e-03,  3.252529827449084e-01,  2.445134137142996e+00,
          3.754408661907416e+00]

    low, high = 0.02425, 1 - 0.02425

    if p < low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    elif p > high:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    else:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)

def auc_to_dprime(auc):
    auc = max(0.5, min(0.999, auc))
    return norm_ppf(auc) * math.sqrt(2)

def run_logic(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        mode = data.get('mode', 'single')

        # 1. Capture base parameters
        ps = 0.5 #float(data.get('Ps', 0.5))
        loa = int(data.get('loa', 5))
        sys_d = auc_to_dprime(float(data.get('auto_auc', 0.75)))
        h_d = auc_to_dprime(float(data.get('human_auc', 0.75)))

        iterations = 2000

        if mode == 'single':
            results = run_simulation(iterations, loa, ps, sys_d, h_d)
            results_obj = {'workload': results.workload, 'accuracy': results.accuracy}
            return JsonResponse({'status': 'success', 'results': results_obj})

        elif mode == '2d':
            param_x = data.get('param_x')
            # Define grid based on the parameter selected
            if param_x == 'loa':
                x_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            # elif param_x in ['auto_auc', 'human_auc']:
            else:
                # Start sensitivity plots at 0.5 as requested
                x_values = np.linspace(0.5, 0.99, 10).tolist()
            # else:  # Ps
            #     x_values = np.linspace(0.01, 0.99, 10).tolist()

            acc_list = []
            wl_list = []

            for x in x_values:
                # Override the specific variable for this loop step
                # curr_ps = x if param_x == 'Ps' else ps
                curr_loa = int(x) if param_x == 'loa' else loa
                curr_sys_auc = x if param_x == 'auto_auc' else sys_d
                curr_hum_auc = x if param_x == 'human_auc' else h_d

                res = run_simulation(
                    iterations,
                    curr_loa,
                    ps,
                    auc_to_dprime(curr_sys_auc),
                    auc_to_dprime(curr_hum_auc)
                )
                acc_list.append(res.accuracy)
                wl_list.append(res.workload)

            plot_data = [
                {'x': x_values, 'y': acc_list, 'name': 'Accuracy', 'line': {'color': '#3498db'}},
                {'x': x_values, 'y': wl_list, 'name': 'Workload', 'line': {'color': '#e67e22'}}
            ]
            return JsonResponse({'status': 'success', 'plot_data': plot_data})

    return JsonResponse({'status': 'error'})