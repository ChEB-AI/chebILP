% Generic Aleph driver for chebILP.
% Usage: swipl -q -g "main('<stem>')" -t halt run_aleph.pl
%   (launched with cwd = the directory holding this file and aleph.pl)
% or pass the stem via the STEM environment variable.

:- initialization(true).

main(Stem) :-
    catch(consult('aleph.pl'), E,
          (print_message(error, E), halt(2))),
    read_all(Stem),
    statistics(walltime, _),
    induce,
    statistics(walltime, [_, MsSince]),
    format("~n=== TIME: ~w ms ===~n", [MsSince]),
    % chebILP: cputime-clock seconds to the best hypothesis, recorded by update_best/7.
    ( '$aleph_global'(time_to_best, TB)
      -> format("~n=== TIME_TO_BEST: ~w s ===~n", [TB])
      ; true ),
    nl,
    write('=== FINAL THEORY ==='), nl,
    ( catch(show(theory), _, fail) -> true ; true ),
    nl,
    halt(0).

main :-
    ( getenv('STEM', Stem) -> main(Stem)
    ; write('no STEM'), nl, halt(1) ).
