import json
import numpy as np
from django.shortcuts import render
from django.http import JsonResponse
from core.sdt import compute_all_ev, Payoffs


def index(request):
    """Renders the main dashboard."""
    return render(request, 'index.html')


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
        s1 = float(data.get('temperature', 1.5))
        s2 = float(data.get('humidity', 1.5))
        a1 = float(data.get('ai1', 1.5))
        a2 = float(data.get('ai2', 1.5))
        human = float(data.get('human', 1.5))

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

        # --- MODE: SINGLE (Metric Cards) ---
        if mode == 'single':
            results = compute_all_ev(**base_params)
            return JsonResponse({'status': 'success', 'results': results})

        # --- MODE: 2D (Sensitivity Line Plot) ---
        elif mode == '2d':
            param_key = data.get('param_x')

            # 1. FIX: Define the range based on the parameter type
            if param_key == "Ps":
                # Probabilities must be between 0 and 1
                x_grid = np.arange(0.05, 0.96, 0.05)
            else:
                # Sensitivities (d') can range from 0 to 5
                x_grid = np.arange(0.5, 5.1, 0.5)

            results_map = {
                'human_two_dss': {'x': [], 'y': [], 'name': 'Human with 2 DSS'}
            }

            for x in x_grid:
                run_params = base_params.copy()
                run_params[param_key] = float(x)

                # 2. RUN MATH
                try:
                    step_res = compute_all_ev(**run_params)
                    ev_value = step_res.get('human_two_dss', 0)
                except:
                    # If the math still fails for a specific value, skip it
                    continue

                results_map['human_two_dss']['x'].append(float(x))
                results_map['human_two_dss']['y'].append(float(ev_value))

            return JsonResponse({
                'status': 'success',
                'plot_data': list(results_map.values()),
                'type': '2d'
            })

        # --- MODE: 3D (Surface Plot) ---
        elif mode == '3d':
            px = data.get('param_x', 'source_1_sensitivity')
            py = data.get('param_y', 'source_2_sensitivity')

            # 1. Setup grids with reasonable resolution for 3D (20-25 points)
            # Ps is 0 to 1, others are 0.5 to 5.0
            x_grid = np.arange(0.1, 0.91, 0.05) if px == "Ps" else np.arange(0.5, 5.1, 0.5)
            y_grid = np.arange(0.1, 0.91, 0.05) if py == "Ps" else np.arange(0.5, 5.1, 0.5)

            z_data = []
            for y_val in y_grid:
                row = []
                for x_val in x_grid:
                    # Create fresh params for this specific coordinate
                    run_params = base_params.copy()
                    run_params[px] = float(x_val)
                    run_params[py] = float(y_val)

                    # 2. Run computation
                    res = compute_all_ev(**run_params)

                    # 3. Target ONLY the 'human_two_dss' key
                    # Using .get() prevents crashes if the key is missing
                    ev_value = res.get('human_two_dss', 0)
                    row.append(float(ev_value))

                z_data.append(row)

            return JsonResponse({
                'status': 'success',
                'plot_data': {
                    'z': z_data,
                    'x': x_grid.tolist(),
                    'y': y_grid.tolist()
                },
                'type': '3d'
            })

    return JsonResponse({'status': 'invalid method'}, status=405)