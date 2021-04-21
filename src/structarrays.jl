const GPUStructArray0{T, N} =
    StructArray{T, N, <: Tuple{Vararg{GPUArrays.AnyGPUArray}}}
const GPUStructArray1{T, N} =
    StructArray{T, N, <: NamedTuple{U, <: Tuple{Vararg{Union{GPUArrays.AnyGPUArray, GPUStructArray0}}}}} where {U}
const GPUStructArray{T, N} = Union{GPUStructArray0{T, N}, GPUStructArray1{T, N}}

# These two similar functions can be removed once
# <https://github.com/JuliaArrays/StructArrays.jl/pull/94>
# is accepted.
function Base.similar(s::StructArray{T,N,C}, ::Type{T}, sz::NTuple{M,Int64}) where {T,N,M,C<:Union{Tuple,NamedTuple}}
    return StructArray{T}(map(typ -> similar(typ, sz),
                              StructArrays.components(s)))
end

function Base.similar(s::StructArray{T,N,C}, S::Type, sz::NTuple{M,Int64}) where {T,N,M,C<:Union{Tuple,NamedTuple}}
    # If not specified, we don't really know what kind of array to use for each
    # interior type, so we just pick the first one arbitrarily. If users need
    # something else, they need to be more specific.
    f1 = StructArrays.components(s)[1]
    if isstructtype(S)
        return StructArrays.buildfromschema(typ -> similar(f1, typ, sz), S)
    else
        return similar(f1, S, sz)
    end
end

# The following broadcast code is slightly modified from the code found at
# <https://github.com/JuliaArrays/StructArrays.jl/issues/150>.
const GPUStore = Tuple{Vararg{GPUArrays.BroadcastGPUArray}}
const NamedGPUStore = NamedTuple{Name,<:GPUStore} where {Name}
const StructGPUArray = StructArray{T,N,<:Union{GPUStore,NamedGPUStore}} where {T,N}
## backend for StructArray
GPUArrays.backend(A::StructGPUArray) =
    GPUArrays.backend(StructArrays.components(A))
GPUArrays.backend(t::GPUStore) = GPUArrays.backend(typeof(t))
GPUArrays.backend(nt::NamedGPUStore) =
    GPUArrays.backend(typeof(nt).parameters[2])
function GPUArrays.backend(::Type{T}) where {T<:GPUStore}
    bs = GPUArrays.backend.(tuple(T.parameters...))
    I = all(map(isequal(first(bs)), bs))
    I || throw("device error")
    GPUArrays.backend(T.parameters[1])
end

## copy from GPUArrays
@inline function Base.copyto!(dest::StructGPUArray,
                              bc::Broadcast.Broadcasted{Nothing})
    axes(dest) == axes(bc) || Broadcast.throwdm(axes(dest), axes(bc))
    isempty(dest) && return dest
    bc′ = Broadcast.preprocess(dest, bc)

    # grid-stride kernel
    function broadcast_kernel(ctx, dest, bc′, nelem)
        for i in 1:nelem
            I = GPUArrays.@cartesianidx(dest, i)
            @inbounds dest[I] = bc′[I]
        end
        return
    end
    heuristic = GPUArrays.launch_heuristic(GPUArrays.backend(dest),
                                           broadcast_kernel, dest, bc′, 1)
    config = GPUArrays.launch_configuration(GPUArrays.backend(dest),
                                            heuristic, length(dest),
                                            typemax(Int))
    GPUArrays.gpu_call(broadcast_kernel, dest, bc′, config.elements_per_thread;
                       threads=config.threads, blocks=config.blocks)

    return dest
end

function Base.similar(bc::Broadcast.Broadcasted{StructArrays.StructArrayStyle{S}},
                      ::Type{T}) where {S<:CUDA.CuArrayStyle,T}
    if isstructtype(T)
        return StructArrays.buildfromschema(typ -> similar(CuArray{typ},
                                                           axes(bc)), T)
    else
        return similar(CuArray{T}, axes(bc))
    end
end

# We follow GPUArrays approach of coping the whole array to the host when
# outputting a StructArray backed by GPU arrays.
convert_to_cpu(xs) = adapt(Array, xs)
function Base.print_array(io::IO, X::GPUStructArray{<:Any,0})
    X = convert_to_cpu(X)
    isassigned(X) ? show(io, X[]) : print(io, undef_ref_str)
end
Base.print_array(io::IO, X::GPUStructArray{<:Any,1}) =
    Base.print_matrix(io, convert_to_cpu(X))
Base.print_array(io::IO, X::GPUStructArray{<:Any,2}) where {T} =
    Base.print_matrix(io, convert_to_cpu(X))
Base.print_array(io::IO, X::GPUStructArray{<:Any,<:Any}) =
    Base.show_nd(io, convert_to_cpu(X), Base.print_matrix, true)

# These definitions allow `StructArray` and `StaticArrays.SArray` to play nicely
# together.
StructArrays.staticschema(::Type{SArray{S,T,N,L}}) where {S,T,N,L} = NTuple{L,T}
StructArrays.createinstance(::Type{SArray{S,T,N,L}}, args...) where {S,T,N,L} =
    SArray{S,T,N,L}(args...)
StructArrays.component(s::SArray, i) = getindex(s, i)

@kernel function fill_kernel!(A, x)
    I = @index(Global)
    @inbounds A[I] = x
end

function Base.fill!(A::GPUStructArray, x)
    event = Event(device(A))
    event = fill_kernel!(device(A), 256)(A, x, ndrange = length(A),
                                         dependencies = (event, ))
    wait(event)
end

function device(s::StructArray)
    ds = map(device, StructArrays.components(s))
    I = all(map(isequal(first(ds)), ds))
    I || throw("device error")
    return first(ds)
end

function Tullio.storage_type(S::StructArray)
    return Tullio.storage_type(StructArrays.components(S)...)
end

components(S::StructArray) = StructArrays.components(S)
