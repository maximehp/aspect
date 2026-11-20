// Positive series coefficients shared by the CPU plane integral and GPU shader.
export function inverseTangentCoefficients(count) {
  const coefficients = new Float64Array(count);
  let area = 1,
    oddHarmonic = 1;
  coefficients[0] = 1;
  for (let i = 1; i < count; i++) {
    area *= (2 * i) / (2 * i + 1);
    oddHarmonic += 1 / (2 * i + 1);
    coefficients[i] = area * oddHarmonic;
  }
  return coefficients;
}
