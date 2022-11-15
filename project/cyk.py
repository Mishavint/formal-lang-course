from pyformlang.cfg import CFG


def cyk(cfg: CFG, str: str):
    if not str:
        return cfg.generate_epsilon()

    n = len(str)
    cnf = cfg.to_normal_form()
    dp = [[set() for _ in range(n)] for _ in range(n)]

    productions_terminal = [
        production for production in cnf.productions if len(production.body) == 1
    ]
    non_productions_terminal = [
        production for production in cnf.productions if len(production.body) == 2
    ]

    for j, symbol in enumerate(str):
        dp[j][j] = set(
            production.head
            for production in productions_terminal
            if production.body[0].value == symbol
        )

    for i in range(1, n):
        for j in range(n - i):
            k = j + i
            for q in range(j, k):
                dp[j][k].update(
                    production.head
                    for production in non_productions_terminal
                    if production.body[0] in dp[j][q]
                    and production.body[1] in dp[q + 1][k]
                )

    return cfg.start_symbol in dp[0][n - 1]
