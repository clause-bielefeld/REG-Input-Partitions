import spacy

class RefExpProcessor:
    """
    RefExpProcessor Class for analyzing captions / referring expressions
    using SpaCy
    """
    def __init__(self, spacy_model, stopwords=None):

        if type(spacy_model) == str:
            self.model = spacy.load(spacy_model)
        else:
            self.model = spacy_model

        self.stopwords = self.model.Defaults.stop_words if not stopwords else stopwords
        self.warnings = []


    def __call__(self, sent):
        return self.model(sent).to_json()['tokens']


    def parse_and_extract_head(
        self, s, return_compound=True
    ):
        """
        Extract the (NP) head of a referring expression;
        return head with data for all tokens in the RefExp.
        """

        # run processor pipeline to get tokens & tags
        dep_struc = self(s)

        # ====================================================================
        # 1) determine the head of the referential NP
        # ====================================================================

        # select root node
        root = [d for d in dep_struc if d["dep"] == "ROOT"]

        if len(root) < 1:  # no root found (e.g. for empty descriptions)

            self.warnings.append(
                {"type": "short", "root": root, "sentence": s, "dep_struc": dep_struc}
            )

            return None, dep_struc  # return None if no head is found

        if len(root) > 1:  # more than one root found (e.g. if space is found)
            self.warnings.append(
                {"type": "long", "root": root, "sentence": s, "dep_struc": dep_struc}
            )
            root = [r for r in root if r["pos"] != "SPACE"]  # sort out "SPACE"
            if len(root) > 1:
                # take the first entry as default
                root = root[0:1]

        # make sure there's only one root candidate left, and unpack it
        assert len(root) == 1, "more than one head token!"
        root = root[0]

        # predicate is head if there's one in the expression
        # in that case: choose the subject as the np head
        if root["pos"] in ["VERB", "AUX"]:
            nsubj = [
                d for d in dep_struc if d["dep"] == "nsubj" and d["head"] == root["id"]
            ]

            if len(nsubj) == 0:
                self.warnings.append(
                    {
                        "type": "no_nsubj_for_verb_root",
                        "root": root,
                        # "root_lemma_in_wn": root["lemma"] in reference,
                        "sentence": s,
                        "dep_struc": dep_struc,
                    }
                )
                nsubj = [root]  # default to root if there are no subjects

            # make sure there's only one nsubj and use it as root
            assert len(nsubj) == 1, str([nsubj, s, dep_struc])
            root = nsubj[0] if len(nsubj) == 1 else root

        # ====================================================================
        # 2) return head depending on the arguments given for return_compound
        # ====================================================================

        if return_compound:  # if compounds shall be returned

            # get compounds
            compounds = [
                d
                for d in dep_struc
                if d["dep"] == "compound" and d["head"] == root["id"]
            ]
            root_and_compounds = compounds + [root]

            # make sure compound nouns are in the correct order
            # (although the compound head should be right in most cases)
            sorted_root_compounds = sorted(
                root_and_compounds, key=lambda x: x.get("start")
            )

            return sorted_root_compounds, dep_struc

        # if not return_compound or len(root_and_compounds) < 2:
        return root, dep_struc