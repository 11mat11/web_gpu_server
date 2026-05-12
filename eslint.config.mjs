import eslint from '@eslint/js';
import tseslint from 'typescript-eslint';

export default tseslint.config(
	eslint.configs.recommended,
	// ZMIANA: Przechodzimy z 'recommended' na 'recommendedTypeChecked'
	// To włącza reguły, które weryfikują m.in. błędy z asynchronicznością
	...tseslint.configs.recommendedTypeChecked,
	{
		languageOptions: {
			parserOptions: {
				project: true,
				tsconfigRootDir: import.meta.dirname,
			},
		},
		rules: {
			// --- TWOJE OBECNE DOBRE PRAKTYKI ---
			'@typescript-eslint/no-explicit-any': 'warn',
			'@typescript-eslint/consistent-type-imports': 'error',
			'no-console': 'off',
			'prefer-const': 'error',
			'no-unused-vars': 'off',
			'@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_' }],
			'no-useless-assignment': 'off',

			// --- NOWE REGULY "SENIOR LEVEL" ---

			// 1. BEZPIECZEŃSTWO ASYNCHRONICZNOŚCI (Kluczowe dla Node.js)
			// Zabrania odpalania funkcji asynchronicznych bez 'await' lub '.catch()'.
			// Zapobiega wyciekom pamięci i cichym błędom (np. unhandled promise rejection).
			'@typescript-eslint/no-floating-promises': 'error',

			// Zabrania pisania 'await' przed czymś, co nie jest Promisem (usuwa martwy kod).
			'@typescript-eslint/await-thenable': 'error',
			'@typescript-eslint/require-await': 'off',

			// 2. ŻELAZNA LOGIKA JAVASCRIPTU
			// Wymusza używanie ścisłego porównania (===) zamiast luźnego (==).
			// Unikasz dzięki temu błędów gdzie np. 0 == false (co jest prawdą w JS, a często błędem logiki).
			eqeqeq: ['error', 'always'],

			// 3. CZYSTOŚĆ I NOWOCZESNOŚĆ
			// Automatycznie skraca obiekty. Zamiast { size: size } wymusza czyste { size }.
			'object-shorthand': 'error',
		},
	},
	{
		ignores: ['build/', 'node_modules/', 'dist/', 'server.key', 'server.cert'],
	},
);
