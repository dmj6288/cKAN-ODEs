function activation_getter_3x3(pM_, pM_new, kan1, grid_size)
    lay1 = kan1[1]
    lay2 = kan1[2]
    st = stM[1]
    
    # Define coefficients for layer 1 and layer 2
    pc1 = pM_new[1].C
    pc2 = pM_new[2].C

    # Split coefficients into separate components for 3D inputs
    pc1x = pc1[:, 1:grid_size]
    pc1y = pc1[:, grid_size+1:2*grid_size]
    pc1z = pc1[:, 2*grid_size+1:end]

    # Define weights for layer 1 and layer 2
    pw1 = pM_new[1].W
    pw2 = pM_new[2].W

    # Split weights for 3D inputs
    pw1x = pw1[:, 1]
    pw1y = pw1[:, 2]
    pw1z = pw1[:, 3]

    size_in = size(X)
    size_out = (lay1.out_dims, size_in[2:end]...)

    x = reshape(X, lay1.in_dims, :)
    K = size(x, 2)

    x_norm = lay1.normalizer(x)
    x_resh = reshape(x_norm, 1, :) 
    basis = lay1.basis_func(x_resh, st.grid, lay1.denominator)

    # Compute activations for each dimension
    activations_x = basis[:, 1:3:end]' * pc1x' .+ (lay1.base_act.(x[1, :]) .* pw1x')
    activations_y = basis[:, 2:3:end]' * pc1y' .+ (lay1.base_act.(x[2, :]) .* pw1y')
    activations_z = basis[:, 3:3:end]' * pc1z' .+ (lay1.base_act.(x[3, :]) .* pw1z')

    ## Second Layer
    LV_samples_lay1 = kan1[1](X, pM_.layer_1, stM[1])[1]

    x = reshape(LV_samples_lay1, lay2.in_dims, :)
    K = size(x, 2)

    x_norm = lay2.normalizer(x)
    x_resh = reshape(x_norm, 1, :) 
    basis = lay2.basis_func(x_resh, st.grid, lay2.denominator)

    activations_second = zeros(lay2.in_dims * 3, K)
    for i in 1:lay2.in_dims
        basis_curr = basis[:, i:lay2.in_dims:end]
        pc_curr = pc2[:, (i-1)*grid_size+1:i*grid_size]
        activations_curr = basis_curr' * pc_curr' .+ (lay2.base_act.(x[i, :]) .* pw2[:, i]')
        activations_second[(i-1)*3+1:i*3, :] = activations_curr'
    end

    return activations_x, activations_y, activations_z, activations_second, LV_samples_lay1, lay2, K
end
