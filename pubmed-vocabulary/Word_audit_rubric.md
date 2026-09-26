# Rubric for the eligibility audit

Each line of the input file is a lowercased word form that occurs in PubMed abstracts. Classify every word form separately (inflections count as separate forms). Judge the word as it is typically used in biomedical and scientific abstracts.

G = General academic wording. The word can be used in abstracts from almost any research field to describe aims, reasoning, general methods, results, comparisons, emphasis, evaluation or text structure, and its meaning does not depend on the specific topic studied. Examples: additionally, notably, findings, highlights, robust, comprehensive, although, whereas, remains, enhance, insights, potential, significant.

T = Topic-specific or technical. The word names, or is mainly used for, a specific research topic, disease, organism, anatomical structure, substance, gene or protein, device, population, setting or policy area, or a field-specific technical, statistical or computational concept. Also: proper names, abbreviations and acronyms, units, and publication metadata or section labels. Examples: ferroptosis, microbiome, nomogram, algorithm, neural, regression, vaccine, hospitalization, telehealth, copyright, registration.

B = Borderline. The word has a common general sense, but in biomedical abstracts it is also often part of a technical or topical term, and both uses are common. Examples: resistance (drug resistance), expression (gene expression), culture (cell culture).

Output format: one line per input word, in input order, as `word,label`.
