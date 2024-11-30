
function activation_getter_3x3(pM_, pM_new, kan1, grid_size)
    lay1 = kan1[1]
    lay2 = kan1[2]
    st = stM[1]
    pc1 = pM_new[1].C
    pc2 = pM_new[2].C
    pc1_split = [pc1[:, (i-1)*grid_size+1:i*grid_size] for i in 1:3]  # Split into 3 parts
    pw1 = pM_new[1].W
    pw1_split = [pw1[:, i] for i in 1:3]

    size_in = size(X)
    size_out = (lay1.out_dims, size_in[2:end]...)

    x = reshape(X, lay1.in_dims, :)
    K = size(x, 2)

    x_norm = lay1.normalizer(x)
    x_resh = reshape(x_norm, 1, :) 
    basis = lay1.basis_func(x_resh, st.grid, lay1.denominator)

    # Compute activations for 3 outputs
    activations_x = zeros(K, size(pc1_split[1], 1))
    activations_y = zeros(K, size(pc1_split[2], 1))
    activations_z = zeros(K, size(pc1_split[3], 1))

    for i in 1:3
        basis_curr = basis[:, i:3:end]
        activations = basis_curr' * pc1_split[i]' .+ (lay1.base_act.(x[i, :]) .* pw1_split[i]')
        if i == 1
            activations_x .= activations
        elseif i == 2
            activations_y .= activations
        elseif i == 3
            activations_z .= activations
        end
    end

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

    return activations_x, activations_y, activations_z, activations_second
end

#Lots of the drivers and plotters have to extract the activations from the KAN-ODE
#for plotting, visualization, pruning, etc. This shared function enables this. 
function activation_getter(pM_, pM_new, kan1, grid_size)
    lay1 = kan1[1]
    lay2 = kan1[2]
    st = stM[1]
    pc1 = pM_new[1].C
    pc2 = pM_new[2].C
    pc1_split = [pc1[:, (i-1)*grid_size+1:i*grid_size] for i in 1:3]  # Split into 3 parts
    pw1 = pM_new[1].W
    pw1_split = [pw1[:, i] for i in 1:3]

    size_in = size(X)
    size_out = (lay1.out_dims, size_in[2:end]...)

    x = reshape(X, lay1.in_dims, :)
    K = size(x, 2)

    x_norm = lay1.normalizer(x)
    x_resh = reshape(x_norm, 1, :) 
    basis = lay1.basis_func(x_resh, st.grid, lay1.denominator)

    # Compute activations for 3 outputs
    activations = [zeros(K) for _ in 1:3]
    for i in 1:3
        basis_curr = basis[:, i:3:end]
        activations[i] = basis_curr' * pc1_split[i]' .+ (lay1.base_act.(x[i, :]) .* pw1_split[i]')
    end

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

    return activations, activations_second, LV_samples_lay1, lay2, K
end
