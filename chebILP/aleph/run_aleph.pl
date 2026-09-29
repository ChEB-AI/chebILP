% Generic Aleph driver for chebILP.
% Usage: swipl -q -g "main('<stem>')" -t halt run_aleph.pl
%   (launched with cwd = the directory holding this file and aleph.pl)
% or pass the stem via the STEM environment variable.

:- initialization(true).

main(Stem) :-
    catch(consult('aleph.pl'), E,
          (print_message(error, E), halt(2))),
    % chebILP: wall clock started after the engine consult but before read_all, so the problem
    % data load is charged to the --timeout budget while the engine load is not -- matching
    % Popper, whose timeout clock starts after `import popper` but before its Tester data load.
    get_time(T0),
    read_all(Stem),
    % chebILP: publish the total wall budget (searchtime) and its start, so aleph.pl stops each
    % search once the budget is spent (discontinue_search) and ends the induce loop with it.
    ( setting(searchtime, Budget), number(Budget)
      -> retractall('$aleph_global'(wall_start, _)),
         retractall('$aleph_global'(wall_budget, _)),
         asserta('$aleph_global'(wall_start, T0)),
         asserta('$aleph_global'(wall_budget, Budget))
      ; true ),
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
