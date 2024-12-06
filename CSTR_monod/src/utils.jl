#
@inline _broadcast(f, args...) = @. f(args...)

#======================================================#
# Radial basis functions (RBF)
#======================================================#

@inline function rbf(x, z, h) # exp(-((x - z)/h)^2)
    y = @. (x - z) * (1/h)
    _rbf(y)
end

@inline _rbf(x) = @. exp(-x^2)

function CRC.rrule(::typeof(_rbf), x)
    T = eltype(x)
    y = _rbf(x)
    @inline ∇_rbf(ȳ) = CRC.NoTangent(), @fastmath(@. -T(2) * x * y * ȳ)

    y, ∇_rbf
end

#======================================================#
# Reflectional SWitch Activation Function (RSWAF)
#======================================================#

@inline function rswaf(x, z, h)
    y = @. (x - z) * (1/h)
    _rswaf(y)
end

@inline function _rswaf(x) # sech(x)^2
    @. 1 - tanh(x)^2
end

function CRC.rrule(::typeof(_rswaf), x)
    T = eltype(x)
    tx = @. tanh(x)
    y  = @. T(1) - tx^2
    @inline ∇_rswaf(ȳ) = CRC.NoTangent(), @fastmath(@. -T(2) * tx * y * ȳ)

    y, ∇_rswaf
end

#======================================================#
# Inverse Quadratic Function (IQF)
#======================================================#

@inline function iqf(x, z, h) # exp(-((x - z)/h)^2)
    y = @. (x - z) * (1/h)
    _iqf(y)
end

@inline _iqf(x) = @. 1 / (1 + x^2)

function CRC.rrule(::typeof(_iqf), x)
    T = eltype(x)
    y = _iqf(x)
    @inline ∇_iqf(ȳ) = CRC.NoTangent(), @fastmath(@. -T(2) * x * y * ȳ)

    y, ∇_iqf
end
#

#======================================================#
# Gaussian Basis Function (Gaussian Basis Function)
#======================================================#

@inline function gaussian_basis_function(x, z, h)
    # Compute Gaussian kernel
    y = @. (x - z) / h # Normalized distance
    _gaussian_basis_function(y)
end

@inline function _gaussian_basis_function(x)
    @. exp(-0.5 * x^2) # Gaussian exponential
end

# Define the reverse-mode gradient for the Gaussian function
function CRC.rrule(::typeof(_gaussian_basis_function), x)
    T = eltype(x)
    y = _gaussian_basis_function(x) # Evaluate Gaussian
    @inline ∇_gaussian_basis_function(ȳ) = CRC.NoTangent(), @fastmath(@. -T(x) * y * ȳ)

    y, ∇_gaussian_basis_function
end
