def get_colors_range(df):
    bananas = df[df['fruit'] == 'banana']
    bananas_r_range = [min(bananas['r']), max(bananas['r'])]
    bananas_g_range = [min(bananas['g']), max(bananas['g'])]
    bananas_b_range = [min(bananas['b']), max(bananas['b'])]

    print('bananas', bananas_r_range, bananas_g_range, bananas_b_range)

    kiwis = df[df['fruit'] == 'kiwi']
    kiwis_r_range = [min(kiwis['r']), max(kiwis['r'])]
    kiwis_g_range = [min(kiwis['g']), max(kiwis['g'])]
    kiwis_b_range = [min(kiwis['b']), max(kiwis['b'])]

    print('kiwis', kiwis_r_range, kiwis_g_range, kiwis_b_range)

    laranjas = df[df['fruit'] == 'laranja']
    laranjas_r_range = [min(laranjas['r']), max(laranjas['r'])]
    laranjas_g_range = [min(laranjas['g']), max(laranjas['g'])]
    laranjas_b_range = [min(laranjas['b']), max(laranjas['b'])]

    print('laranjas', laranjas_r_range, laranjas_g_range, laranjas_b_range)

    macas = df[df['fruit'] == 'maca']
    macas_r_range = [min(macas['r']), max(macas['r'])]
    macas_g_range = [min(macas['g']), max(macas['g'])]
    macas_b_range = [min(macas['b']), max(macas['b'])]

    print('macas', macas_r_range, macas_g_range, macas_b_range)

    peras = df[df['fruit'] == 'pera']
    peras_r_range = [min(peras['r']), max(peras['r'])]
    peras_g_range = [min(peras['g']), max(peras['g'])]
    peras_b_range = [min(peras['b']), max(peras['b'])]
    
    print('peras', peras_r_range, peras_g_range, peras_b_range)
