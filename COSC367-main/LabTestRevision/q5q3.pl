tran(eins,one).
tran(zwei,two).
tran(drei,three).
tran(vier,four).
tran(fuenf,five).
tran(sechs,six).
tran(sieben,seven).
tran(acht,eight).
tran(neun,nine).

listtran([], []).
listtran([A|L1], [B|L2]) :- tran(A,B), listtran(L1, L2).

test_answer :-
    listtran([eins, neun, zwei], X),
    writeln(X).

test_answer1 :-
    listtran([], []),
    writeln('OK').


test_answer2 :-
    listtran(X, [one, seven, six, two]),
    writeln(X).

test_answer3 :-
    listtran(L1, L2),
    writeln('OK').


tran(1, one).
tran(2, two).
tran(3, three).

test_answer4 :-
    listtran([1, 2, 3], X),
    writeln(X).
