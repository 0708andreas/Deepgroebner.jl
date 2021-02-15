using Flux
using ReinforcementLearning


struct SMParallel{T}
    layers::T


struct Parallel{F, T}
  connection::F
  layers::T
end

Parallel(connection, layers...) = Parallel(connection, layers)

@functor Parallel

(m::Parallel)(x::AbstractArray) = mapreduce(f -> f(x), m.connection, m.layers)
(m::Parallel)(xs::Vararg{<:AbstractArray}) = mapreduce((f, x) -> f(x), m.connection, m.layers, xs)
(m::Parallel)(xs::Tuple) = m(xs...)

Base.getindex(m::Parallel, i::Integer) = m.layers[i]
Base.getindex(m::Parallel, i::AbstractVector) = Parallel(m.connection, m.layers[i]...)

function Base.show(io::IO, m::Parallel)
  print(io, "Parallel(", m.connection, ", ")
  join(io, m.layers, ", ")
  print(io, ")")
end
