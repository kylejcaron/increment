"""High-precision primitive OLS for identical integer covariates in both arms."""

from decimal import Decimal, localcontext


def balanced_adjusted_reference(values):
    with localcontext() as ctx:
        ctx.prec = 70
        n = len(values) // 2
        x = [Decimal(i) - Decimal(n - 1) / 2 for i in range(n)]
        xx = sum((a * a for a in x), Decimal(0))
        means: list[Decimal] = []
        rss = Decimal(0)
        for arm in range(2):
            y = [Decimal.from_float(float(v)) for v in values[arm * n : (arm + 1) * n]]
            mean = sum(y, Decimal(0)) / n
            slope = sum((a * b for a, b in zip(x, y, strict=True)), Decimal(0)) / xx
            means.append(mean)
            rss += sum(((b - mean - slope * a) ** 2 for a, b in zip(x, y, strict=True)), Decimal(0))
        variance = Decimal(2 * n) / Decimal(2 * n - 1) * rss / Decimal(n * n)
        return float(means[1] - means[0]), float(variance.sqrt())
