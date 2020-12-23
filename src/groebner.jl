import Base.:*, Base.:>, Base.:<
# A term is a pair of coefficient

struct term{N, T}
    l :: T
    a :: NTuple{N, Int64}
end

# Grevlex order
# 
# a > b if total (sum?) degree of a > b or,
# if equal the last non-zero term of a - b is negative
function gt(a, b)
    if     (sum(a) > sum(b)) return true
    elseif (sum(a) < sum(b)) return false
    else   return filter(!=(0), a .- b)[end] < 0 end
end

>(a :: term, b :: term) = gt(a.a, b.a) 
<(a :: term, b :: term) = not(gt(a.a, b.a))

# Term division
(|)(t :: term, r :: term) = all(t.a .<= r.a) # Ja, std <=, ikke grevlex ;)
div = (|)
(/)(t :: term, r :: term) = term(t.l/r.l, t.a .- r.a)

# Term lcm
lcm(t :: term, r :: term) = term(one(t.l), max.(t.a, r.a))

# Polynomial-term multiplication
(*)(t :: term, r :: term) = term(t.l * r.l, t.a .+ r.a)
(*)(t :: term, f ) = Ref(t) .* f

# Polynomial subtraction


# Leading term
# 
# (lambda, a) at the largest a according to grevlex
LT(f) = reduce((a, b) -> if (gt(a.a, b.a)) a else b end, f)

S(f, g) = let gamma = lcm(LT(f), LT(g)).a
    (term(1, gamma)/LT(f))*f - (term(1, gamma)/LT(g))*g
end

# Multivariate division
# function mdiv(h, F)
#     r = h
#     while any(LT.(F) .| Ref(LT(r)))
#         i = findfirst(LT.(F) .| Ref(LT(r)))
#         f = F[i]
#         r =

# Update function; TODO: refine this
# BEWARE: mutation!!!
update(P, G, r) = append!(P, [(f, r) for f = G])
select(P) = p[1]

# function buchberger(G)
#     P = [(f, g) for f = G, g = G]
#     while length(P) > 0
#         (f, g) = pop!(P)
