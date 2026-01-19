import json
import sys
import numpy as np
from django.shortcuts import render
from django.http import JsonResponse
from core.sdt import compute_all_ev, Payoffs
import math


def index(request):
    """Renders the main dashboard."""
    return render(request, 'index.html')

# def auc_to_dprime(auc):
#     auc = max(0.5, min(0.999, auc))
#     return norm.ppf(auc) * math.sqrt(2)

def norm_ppf(p):
    """
    High-precision Inverse Normal CDF for Python < 3.12.
    Uses the relationship: ppf(p) = sqrt(2) * erfinv(2p - 1)
    Approximated via the high-accuracy Acklam's method.
    """
    if p <= 0 or p >= 1:
        return float('nan')

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
    """
    Handles Single Update, 2D Sensitivity Plots, and 3D Surface Plots.
    """
    if request.method == 'POST':
        data = json.loads(request.body)
        mode = data.get('mode', 'single')

        # 1. Capture current baseline values from the diagram sliders
        # These are used as 'constants' for the parameters NOT being varied
        print(data)
        ps_val = float(data.get('Ps', 0.2))
        auc_temp = float(data.get('temperature', 0.75))
        auc_humid = float(data.get('humidity', 0.75))
        auc_ai1 = float(data.get('ai1', 0.75))
        auc_ai2 = float(data.get('ai2', 0.75))

        # Convert AUC to d' (Sensitivity)
        s1 = auc_to_dprime(auc_temp)
        s2 = auc_to_dprime(auc_humid)
        a1 = auc_to_dprime(auc_ai1)
        a2 = auc_to_dprime(auc_ai2)

        payoffs = Payoffs(
            V_TP=float(data.get('v_tp', 1)),
            V_FP=float(data.get('v_fp', -1)),
            V_FN=float(data.get('v_fn', -2)),
            V_TN=float(data.get('v_tn', 1))
        )

        # 3. Capture AI Costs
        c_ai1 = float(data.get('c_ai1', 0))
        c_ai2 = float(data.get('c_ai2', 0))

        # Define base parameters dictionary
        base_params = {
            "Ps": ps_val,
            "source_1_sensitivity": s1,
            "source_2_sensitivity": s2,
            "DSS1_sensitivity": a1,
            "DSS2_sensitivity": a2,
            "payoffs": payoffs,
            "DSS1_cost": c_ai1,
            "DSS2_cost": c_ai2
        }

        print(base_params)

        # --- MODE: SINGLE (Metric Cards) ---
        if mode == 'single':
            results = compute_all_ev(**base_params)
            return JsonResponse({'status': 'success', 'results': results})

        # --- MODE: 2D (Sensitivity Line Plot) ---
        elif mode == '2d':
            param_key = data.get('param_x')

            x_grid = np.arange(0.05, 0.96, 0.05) if param_key == "Ps" else np.arange(0.55, 0.96, 0.05)

            results_map = {
                'human_two_dss': {'x': [], 'y': [], 'name': 'Human + 2 AI', 'type': 'scatter', 'mode': 'lines+markers'}}

            for x in x_grid:
                run_params = base_params.copy()
                run_params[param_key] = float(x) if param_key == "Ps" else auc_to_dprime(x)
                try:
                    step_res = compute_all_ev(**run_params)
                    results_map['human_two_dss']['x'].append(float(x))
                    results_map['human_two_dss']['y'].append(float(step_res.get('human_two_dss', 0)))
                except Exception as e:
                    continue

            return JsonResponse({
                'status': 'success',
                'plot_data': list(results_map.values()),
                'type': '2d'
            })

        # --- MODE: 3D (Surface Plot) ---
        elif mode == '3d':
            px, py = data.get('param_x'), data.get('param_y')
            x_grid = np.arange(0.1, 0.95, 0.05) if px == "Ps" else np.arange(0.55, 0.96, 0.05)
            y_grid = np.arange(0.1, 0.95, 0.05) if py == "Ps" else np.arange(0.55, 0.96, 0.05)

            z_data = []
            for y_val in y_grid:
                row = []
                for x_val in x_grid:
                    run_params = base_params.copy()
                    run_params[px] = float(x_val) if px == "Ps" else auc_to_dprime(x_val)
                    run_params[py] = float(y_val) if py == "Ps" else auc_to_dprime(y_val)
                    res = compute_all_ev(**run_params)
                    row.append(float(res.get('human_two_dss', 0)))
                z_data.append(row)

            return JsonResponse(
                {'status': 'success', 'plot_data': {'z': z_data, 'x': x_grid.tolist(), 'y': y_grid.tolist()},
                 'type': '3d'})

    return JsonResponse({'status': 'invalid method'}, status=405)