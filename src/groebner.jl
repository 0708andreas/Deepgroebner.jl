import Base.:*, Base.:>, Base.:<

export term,
    gt,
    |,
    /,
    div,
    *,
    lcm,
    minus,
    LT,
    S,
    mdiv,
    buchberger,
    is_groebner_basis

# A term is a pair of coefficient (Float64) and exponent
struct term{N}
    l :: Float64
    a :: NTuple{N, Int64}
end

binom(t :: term) = [t, term(0.0, ntuple(x->0, length(t.a)))]

# Grevlex order
# 
# a > b if total (sum?) degree of a > b or,
# if equal the last non-zero term of a - b is negative
function gt(a, b)
    if a == b                return false
    elseif (sum(a) > sum(b)) return true
    elseif (sum(a) < sum(b)) return false
    else   return filter(!=(0), a .- b)[end] < 0 end
end

>(a :: term, b :: term) = gt(a.a, b.a) 
<(a :: term, b :: term) = not(gt(a.a, b.a))

# Term division
(|)(t :: term, r :: term) = all(t.a .<= r.a) # Ja, std <=, ikke grevlex ;)
div = (|)
(/)(t :: term, r :: term) = term(Base.:/(t.l, r.l), t.a .- r.a)

# Term lcm
lcm(t :: term, r :: term) = term(one(t.l), max.(t.a, r.a))

# Polynomial-term multiplication
(*)(t :: term, r :: term) = term(t.l * r.l, t.a .+ r.a)
(*)(t :: term, f ) = Ref(t) .* f

# Polynomial subtraction
minus(f, g) = let d = merge(+, Dict(x.a => x.l for x in f),
                              Dict(x.a => -1*x.l for x in g))
    [term(v, k) for (k, v) in d if v != 0]
end


# Leading term
# 
# (lambda, a) at the largest a according to grevlex
LT(f) = reduce((a, b) -> if (gt(a.a, b.a)) a else b end, f)

# S(f, g) = let gamma = lcm(LT(f), LT(g)).a
#     ((term(1, gamma)/LT(f))*f) - ((term(1, gamma)/LT(g))*g)
# end

S(f, g) = let gamma = lcm(LT(f), LT(g))
    minus((gamma/LT(f))*f, (gamma/LT(g))*g)
end


# Multivariate division

function mdiv_count(h, F)
    r = h
    count = 0
    while r != [] && any(LT.(F) .| Ref(LT(r)))
        i = findfirst(LT.(F) .| Ref(LT(r)))
        f = F[i]
        r = minus(r, (LT(r)/LT(f))*f)
        count = count + 1
        # println(r)
    end
    return (r, count)
end

mdiv(h, F) = mdiv_count(h, F)[1]

# reduce = mdiv


# Update function; TODO: refine this
# BEWARE: mutation!!!
# update!(P, G, r) = append!(P, [(f, r) for f in G])

function update!(P, G, r)
    m = length(G)
    ltG = LT.(G)
    ltr = LT(r)
    function can_drop(p)
        f, g = p
        gam = lcm(LT(f), LT(g))
        return (ltr | gam) && gam != lcm(LT(f), ltr) && gam != lcm(LT(g), ltr)
    end
    filter!(p -> !can_drop(p), P)

    # println("filtered")
    lcms = Dict{term, Vector{Vector{term}}}()
    for f in G
        if lcm(LT(f), ltr) in keys(lcms)
            push!(lcms[lcm(LT(f), ltr)], f)
        else
            lcms[lcm(LT(f), ltr)] = [f]
        end
    end

    # println("computed lcms")
    # println(length(keys(lcms)))
    min_lcms = []
    P_ = []
    for gam in sort(collect(keys(lcms)), lt = ((x, y) -> ! gt(x.a, y.a)))
    # for gam in keys(lcms)
        if all([! (gam | t) for t in min_lcms])
            push!(min_lcms, gam)
            if ! any([lcm(LT(t), ltr) == LT(t) * ltr for t in lcms[gam] ])
                push!(P_, (lcms[gam][1], r))
            end
        end
    end
    append!(P, P_)
end


select!(P) = pop!(P)

function buchberger(G)
    P = [(f, g) for f in G for g in G]
    t = 0
    while length(P) > 0
        # println("start")
        (f, g) = select!(P)
        # println("selected")
        r = mdiv(S(f, g), G)
        # println("division done")
        if r != []
            P = update!(P, G, r)
            G = append!(G, [r])
        end
        t += 1
        # println(t)
        # println(length(P))

    end
    return G
end

function is_groebner_basis(G)
    return all([mdiv(S(G[i], G[j]), G) == [] for i in 1:length(G) for j in i:length(G)])
end
