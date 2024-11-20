twice([],[]).
twice([H1|T1],[H1, H1| T2]) :- twice(T1, T2).

test_answer0 :-
    twice([a, b, c, d], L),
    writeln(L).

test_answer1 :-
    twice(L, [1, 1, 2, 2, 3, 3]),
    writeln(L).

test_answer2 :-
    twice([], []),
    writeln('OK').

test_answer3 :-
    twice(L1, L2),
    writeln('OK').

test_answer4 :-
    \+ twice(L, [a, a, b]),
    writeln('OK').
