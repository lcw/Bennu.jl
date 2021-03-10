"""
    minmaxflip(x, y)

Returns `x, y` sorted lowest to highest and a bool that indicates if a swap
was needed.
"""
minmaxflip(x, y) = y < x ? (y, x, true) : (x, y, false)

permutationtuples(n) = permutationtuples(Val(n))
permutationtuples(::Val{1}) = ((1,),)
permutationtuples(::Val{2}) = ((1,2),(2,1))
permutationtuples(::Val{3}) = ((1,2,3),(1,3,2),(2,1,3),(2,3,1),(3,1,2),(3,2,1))
permutationtuples(::Val{4}) = ((1,2,3,4), (1,2,4,3), (1,3,2,4), (1,3,4,2),
                               (1,4,2,3), (1,4,3,2), (2,1,3,4), (2,1,4,3),
                               (2,3,1,4), (2,3,4,1), (2,4,1,3), (2,4,3,1),
                               (3,1,2,4), (3,1,4,2), (3,2,1,4), (3,2,4,1),
                               (3,4,1,2), (3,4,2,1), (4,1,2,3), (4,1,3,2),
                               (4,2,1,3), (4,2,3,1), (4,3,1,2), (4,3,2,1))
"""
    tuplesort(a)

Returns the tuple `a` as a sorted tuple.
"""
@inline tuplesort(a::Tuple) = tuplesort(a...)

"""
    tuplesortpermnum(a)

Returns the lexicographic permutation number `p` to sort the tuple `a`.
"""
@inline tuplesortpermnum(a::Tuple) = tuplesortpermnum(a...)

"""
    tuplesortperm(a)

Returns the permutation to sort `a`.
"""
function tuplesortperm(a::Tuple)
    return @inbounds permutationtuples(length(a))[tuplesortpermnum(a)]
end

tuplesortpermutation(x::Tuple) = Permutation{length(x)}(tuplesortpermnum(x...))
tuplesortpermutation(args...) = Permutation{length(args)}(tuplesortpermnum(args...))

tuplesort(a) = Tuple(a)
tuplesortpermnum(a) = 1

tuplesort(a, b) = minmax(a, b)
function tuplesortpermnum(a, b)
    a, b, s1 = minmaxflip(a, b)
    p = s1 ? 2 : 1
    return p
end

function tuplesort(a, b, c)
    # Use a (Bose-Nelson Algorithm based) sorting network from
    # <http://pages.ripco.net/~jgamble/nw.html> to sort the vertices.
    b, c = minmax(b, c)
    a, c = minmax(a, c)
    a, b = minmax(a, b)

    return (a, b, c)
end

function tuplesortpermnum(a, b, c)
    # Use a (Bose-Nelson Algorithm based) sorting network from
    # <http://pages.ripco.net/~jgamble/nw.html> to sort the vertices.
    b, c, s1 = minmaxflip(b, c)
    a, c, s2 = minmaxflip(a, c)
    a, b, s3 = minmaxflip(a, b)

    if     !s1 && !s2 && !s3; return 1
    elseif  s1 && !s2 && !s3; return 2
    elseif !s1 &&  s2 && !s3; return 4
    elseif  s1 &&  s2 && !s3
        @assert c < b && b ≤ c
        throw(AssertionError())
    elseif !s1 && !s2 &&  s3; return 3
    elseif  s1 && !s2 &&  s3; return 5
    elseif !s1 &&  s2 &&  s3; return 4
    elseif  s1 &&  s2 &&  s3; return 6
    else
        # Throw an error as this should never be reached.
        throw(AssertionError())
    end
end

function tuplesort(a, b, c, d)
    # Use a (Bose-Nelson Algorithm based) sorting network from
    # <http://pages.ripco.net/~jgamble/nw.html> to sort the vertices.
    a, b = minmax(a, b)
    c, d = minmax(c, d)
    a, c = minmax(a, c)
    b, d = minmax(b, d)
    b, c = minmax(b, c)

    return (a, b, c, d)
end

function tuplesortpermnum(a, b, c, d)
    # Use a (Bose-Nelson Algorithm based) sorting network from
    # <http://pages.ripco.net/~jgamble/nw.html> to sort the vertices.
    a, b, s1 = minmaxflip(a, b)
    c, d, s2 = minmaxflip(c, d)
    a, c, s3 = minmaxflip(a, c)
    b, d, s4 = minmaxflip(b, d)
    b, c, s5 = minmaxflip(b, c)

    if     !s1 && !s2 && !s3 && !s4 && !s5; return 1
    elseif  s1 && !s2 && !s3 && !s4 && !s5; return 7
    elseif !s1 &&  s2 && !s3 && !s4 && !s5; return 2
    elseif  s1 &&  s2 && !s3 && !s4 && !s5; return 8
    elseif !s1 && !s2 &&  s3 && !s4 && !s5; return 15
    elseif  s1 && !s2 &&  s3 && !s4 && !s5; return 13
    elseif !s1 &&  s2 &&  s3 && !s4 && !s5; return 19
    elseif  s1 &&  s2 &&  s3 && !s4 && !s5
        @assert b < a && d < c && d < b && a ≤ c && a ≤ b
        throw(AssertionError())
    elseif !s1 && !s2 && !s3 &&  s4 && !s5; return 4
    elseif  s1 && !s2 && !s3 &&  s4 && !s5; return 10
    elseif !s1 &&  s2 && !s3 &&  s4 && !s5
        @assert a ≤ b && d < c && a ≤ d && c < b && c ≤ d
        throw(AssertionError())
    elseif  s1 &&  s2 && !s3 &&  s4 && !s5
        @assert b < a && d < c && b ≤ d && c < a && c ≤ d
        throw(AssertionError())
    elseif !s1 && !s2 &&  s3 &&  s4 && !s5; return 17
    elseif  s1 && !s2 &&  s3 &&  s4 && !s5; return 18
    elseif !s1 &&  s2 &&  s3 &&  s4 && !s5; return 23
    elseif  s1 &&  s2 &&  s3 &&  s4 && !s5; return 24
    elseif !s1 && !s2 && !s3 && !s4 &&  s5; return 3
    elseif  s1 && !s2 && !s3 && !s4 &&  s5; return 9
    elseif !s1 &&  s2 && !s3 && !s4 &&  s5; return 5
    elseif  s1 &&  s2 && !s3 && !s4 &&  s5; return 11
    elseif !s1 && !s2 &&  s3 && !s4 &&  s5; return 13
    elseif  s1 && !s2 &&  s3 && !s4 &&  s5; return 15
    elseif !s1 &&  s2 &&  s3 && !s4 &&  s5; return 19
    elseif  s1 &&  s2 &&  s3 && !s4 &&  s5; return 21
    elseif !s1 && !s2 && !s3 &&  s4 &&  s5; return 4
    elseif  s1 && !s2 && !s3 &&  s4 &&  s5; return 10
    elseif !s1 &&  s2 && !s3 &&  s4 &&  s5; return 6
    elseif  s1 &&  s2 && !s3 &&  s4 &&  s5; return 12
    elseif !s1 && !s2 &&  s3 &&  s4 &&  s5; return 14
    elseif  s1 && !s2 &&  s3 &&  s4 &&  s5; return 16
    elseif !s1 &&  s2 &&  s3 &&  s4 &&  s5; return 20
    elseif  s1 &&  s2 &&  s3 &&  s4 &&  s5; return 22
    else
        # Throw an error as this should never be reached.
        throw(AssertionError())
    end
end
