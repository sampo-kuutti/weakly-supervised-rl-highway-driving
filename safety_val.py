def safety_cage(t, v_rel, t_h, v, x_rel, a, y, k):
    y_1 = 1
    y_2 = 1
    y_3 = 1
    y_4 = 1
    y_5 = 1
    y_6 = 1

    a_comf = 1.67   # desired comfort limit for deceleration
    if x_rel != 0:
        b_k = ((v**2)/(2 * (x_rel - 5)))
    else:
        b_k = a
    ei = b_k / a_comf  # emergency indicator, if ei > 1, the vehicle is in an emergency scenario
    b_emerg = -ei * b_k

    #b_int = ((v * v_rel) ** 2) / (4 * a_comf * (x_rel ** 2))  # IDM braking at high approach rates
    if b_k != 0:
        ei_s = abs(a / b_k)  # emergency indicator for stationary objects ahead
    else:
        ei_s = 1

    # safety cages

    # Rule 1: Stationary object ahead
    if abs(v_rel) < 1.02 * v and abs(v_rel) > 0.98 * v and v > 4.2:
        print('stationary object ahead, x_rel = %f' % x_rel)
        if ei_s < 1:
            y_1 = ei_s - 1
        else:
            y_1 = 0
        print('y = %f, y1 = %f, @ t = %f' % (y, y_1, t))

    # Rule 2: No acceleration at moderate time headway and negative v_rel
    if t_h < 0.91*2 and v_rel < -2:
        y_2 = 0
        print('y = %f, y2 = %f, @ t = %f' % (y, y_2, t))

    # Rule 3: Breaking at low time headway
    if t_h < 0.7*2:
        y_3 = (t_h - 2) / 2.5
        print('y = %f, y3 = %f, @ t = %f' % (y, y_3, t))

    # Rule 4: Large difference in relative velocities
    if v_rel < -3.0 and t_h < 2:
        y_4 = v_rel / 20
        print('y = %f, y4 = %f, @ t = %f' % (y, y_4, t))

    # Rule 5: Critical inter-vehicular distance
    if x_rel < 5:
        y_5 = (x_rel / 10) - 1
        print('y = %f, y5 = %f, @ t = %f' % (y, y_5, t))

    # choose the most conservative action
    output = min([y, y_1, y_2, y_3, y_4, y_5, y_6])

    return output

def safety_cage2(t, v_rel, t_h, v, x_rel, a, y, k):
    y_1 = 1
    y_2 = 1
    y_3 = 1
    y_4 = 1
    y_5 = 1
    y_6 = 1

    #ttc calculation
    if v_rel < 0:
        ttc = x_rel / (-v_rel)
    else:
        ttc = 15    # set ttc as 15s if v_rel >= 0


    # Rule 3: Breaking at low time headway
    if t_h < 1.6 and t_h > 1:
        y_3 = (t_h - 2) / 2
        print('y = %f, y3 = %f, @ t = %f' % (y, y_3, t))
    elif t_h < 1 and t_h > 0.5:
        y_3 = t_h - 1.5
        print('y = %f, y3 = %f, @ t = %f' % (y, y_3, t))
    elif t_h <= 0.5:
        y_3 = -1
        print('y = %f, y3 = %f, @ t = %f' % (y, y_3, t))

    # Rule 4: Large difference in relative velocities
    if ttc < 2.5 and ttc > 1.5:
        y_4 = (ttc - 2.5)/2
        print('y = %f, y4 = %f, @ t = %f' % (y, y_4, t))
    elif ttc < 1.5 and ttc > 1:
        y_4 = ttc - 2
        print('y = %f, y4 = %f, @ t = %f' % (y, y_4, t))
    elif ttc < 1:
        y_4 = -1
        print('y = %f, y4 = %f, @ t = %f' % (y, y_4, t))

    # choose the most conservative action
    arr_output = [y, y_1, y_2, y_3, y_4, y_5, y_6]
    output = min(arr_output)
    sc = arr_output.index(output)

    return sc, output
