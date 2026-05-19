#!/usr/bin/env python3
"""
Convertisseur OWL enrichi -> JSON V2
"""

import re
import json
import html
import unicodedata
from collections import defaultdict
from pathlib import Path

BASE_IRI = "http://webprotege.stanford.edu/"

OBJECT_PROPERTIES = {
    "R7w5XngTituGN8Nt6R834WB": "requires",
    "RC1nx89OA7XMKZ0L1UMLNYg": "supports",
    "RCYrSiiYt2sTA1qKFTVbXbA": "has_qualifiers",
    "R8vQ8mX6hV4s7wN4eSTlvaF": "has_qualifier_families",
    "Rgkbf3QYLEo9sJtKMJFyFW":  "excludes",
    "Rwqzfm396oP2bT07RfnCy8":  "excludes_families",
    "R91SX26q028zwTknzSKDZUj": "weight",
    "R8EpeA2cxOPJQ7nwwuht2D2": "origin_structure",
    "R86MFl68gsSAS3kHPEgghC3": "territory",
    "RBAc9OvdrWdtL7GDUKcc90J": "ecg_morphology",
    "RBNXrhQkzAvi9hGX9yqhyRF": "electrode",
    "R7w87zMopmaSccl0Uko64fb": "origin_territory",
}

ANNOTATION_HIDE = "RBFYm0Sfy9WXu0RwbC5lPp1"
ANNOTATION_IMPORTANCE = "RBiXCmVuqDW3Kzzg8N1v6i3"
ANNOTATION_MAYHAVETERR = "RvQtNXH9Cp7Ss5k9ocYaZD"
ANNOTATION_ACRONYM = "RCytyQcQZm8qtKbCTYQXDB4"
ANNOTATION_MAYHAVEMIRROR = "R81WX84pmfiju3JOXA5ub0A"

POIDS_MAP = {"Urgent": 5, "majeur": 4, "moyen": 3, "descriptif": 2}


def label_to_key(label):
    label = html.unescape(label)
    nfkd = unicodedata.normalize("NFKD", label)
    ascii_str = "".join(c for c in nfkd if not unicodedata.combining(c))
    key = ascii_str.upper()
    key = re.sub(r"['\s\-/,.:;()]+", "_", key)
    key = re.sub(r"_+", "_", key)
    key = key.strip("_")
    key = re.sub(r"[^A-Z0-9_]", "", key)
    return key


def get_all_descendants(iri, child_map, visited=None):
    if visited is None:
        visited = set()
    if iri in visited:
        return set()
    visited.add(iri)
    result = {iri}
    for c in child_map.get(iri, []):
        result |= get_all_descendants(c, child_map, visited)
    return result


def parse_owl(owl_path):
    owl = Path(owl_path).read_text(encoding="utf-8")
    class_pattern = r'<owl:Class rdf:about="http://webprotege\.stanford\.edu/([^"]+)">(.*?)</owl:Class>'
    classes_raw = re.findall(class_pattern, owl, re.DOTALL)

    iri_to_label = {}
    iri_to_label_en = {}
    for iri, body in classes_raw:
        label_fr = re.search(r'<rdfs:label xml:lang="fr">([^<]+)', body)
        label_plain = re.search(r'<rdfs:label>([^<]+)', body)
        label_en = re.search(r'<rdfs:label xml:lang="en">([^<]+)', body)
        lbl = label_fr or label_plain or label_en
        iri_to_label[iri] = html.unescape(lbl.group(1)) if lbl else iri
        if label_en:
            iri_to_label_en[iri] = html.unescape(label_en.group(1))

    parent_map = defaultdict(list)
    child_map = defaultdict(list)
    for iri, body in classes_raw:
        parents = re.findall(r'<rdfs:subClassOf rdf:resource="http://webprotege\.stanford\.edu/([^"]+)"', body)
        for p in parents:
            parent_map[iri].append(p)
            child_map[p].append(iri)

    restriction_pattern = (
        r'<owl:Restriction>\s*'
        r'<owl:onProperty rdf:resource="http://webprotege\.stanford\.edu/([^"]+)"/>\s*'
        r'<owl:someValuesFrom rdf:resource="http://webprotege\.stanford\.edu/([^"]+)"/>'
    )
    class_restrictions = defaultdict(lambda: defaultdict(list))
    for iri, body in classes_raw:
        for prop_iri, target_iri in re.findall(restriction_pattern, body):
            prop_name = OBJECT_PROPERTIES.get(prop_iri)
            if prop_name:
                class_restrictions[iri][prop_name].append(target_iri)

    class_synonymes = defaultdict(list)
    for iri, body in classes_raw:
        syns = re.findall(r'<skos:altLabel[^>]*>([^<]+)', body)
        class_synonymes[iri] = [html.unescape(s) for s in syns]

    class_annotations = defaultdict(dict)
    for iri, body in classes_raw:
        hide_m = re.search(rf'<webprotege:{ANNOTATION_HIDE}[^>]*>(\d+)<', body)
        if hide_m:
            class_annotations[iri]["hide"] = int(hide_m.group(1))
        terr_m = re.search(rf'<webprotege:{ANNOTATION_MAYHAVETERR}[^>]*>(true|false)<', body)
        if terr_m:
            class_annotations[iri]["mayhaveterritory"] = terr_m.group(1) == "true"
        imp_m = re.search(rf'<webprotege:{ANNOTATION_IMPORTANCE}>([^<]+)<', body)
        if imp_m:
            class_annotations[iri]["importance_territoire"] = imp_m.group(1)
        acr_m = re.search(rf'<webprotege:{ANNOTATION_ACRONYM}>([^<]+)<', body)
        if acr_m:
            class_annotations[iri]["acronym"] = html.unescape(acr_m.group(1))
        mir_m = re.search(rf'<webprotege:{ANNOTATION_MAYHAVEMIRROR}>([^<]+)<', body)
        if mir_m:
            class_annotations[iri]["mayhavemirror"] = html.unescape(mir_m.group(1))

    return {
        "classes_raw": classes_raw,
        "iri_to_label": iri_to_label,
        "iri_to_label_en": iri_to_label_en,
        "parent_map": dict(parent_map),
        "child_map": dict(child_map),
        "class_restrictions": {k: dict(v) for k, v in class_restrictions.items()},
        "class_synonymes": dict(class_synonymes),
        "class_annotations": dict(class_annotations),
    }


def determine_types(parsed):
    iri_to_label = parsed["iri_to_label"]
    parent_map = parsed["parent_map"]
    child_map = parsed["child_map"]
    restrictions = parsed["class_restrictions"]
    all_iris = set(iri_to_label.keys())

    top_level = {}
    for iri in all_iris:
        parents_in_onto = [p for p in parent_map.get(iri, []) if p in all_iris]
        if not parents_in_onto:
            top_level[iri] = iri_to_label[iri]

    family_for_iri = {}
    for root_iri, root_label in top_level.items():
        for d in get_all_descendants(root_iri, child_map):
            family_for_iri[d] = root_label

    qualifier_targets = set()
    qualifier_family_targets = set()
    for iri, rels in restrictions.items():
        for t in rels.get("has_qualifiers", []):
            qualifier_targets.add(t)
        for t in rels.get("has_qualifier_families", []):
            qualifier_family_targets.add(t)

    qualifier_iris = set()
    for qt in qualifier_targets:
        qualifier_iris |= get_all_descendants(qt, child_map)
    for qft in qualifier_family_targets:
        qualifier_iris |= get_all_descendants(qft, child_map)

    pattern_iris = set()
    for iri, rels in restrictions.items():
        if "requires" in rels:
            pattern_iris.add(iri)
    expanded = set()
    for p in pattern_iris:
        expanded |= get_all_descendants(p, child_map)
    pattern_iris |= expanded

    REFERENCE_FAMILIES = {"Poids", "Anatomie"}
    # Concepts with their own "requires" restriction are always patterns,
    # even if they are also targets of has_qualifiers from other concepts.
    own_requires = {iri for iri, rels in restrictions.items() if "requires" in rels}

    concept_types = {}
    for iri in all_iris:
        family = family_for_iri.get(iri, "")
        if family in REFERENCE_FAMILIES:
            concept_types[iri] = "reference"
        elif "rivations" in family:
            concept_types[iri] = "reference"
        elif family == "Topographie":
            concept_types[iri] = "topography"
        elif iri in own_requires:
            concept_types[iri] = "pattern"
        elif iri in pattern_iris:
            concept_types[iri] = "pattern"
        elif iri in qualifier_iris:
            concept_types[iri] = "qualifier"
        else:
            concept_types[iri] = "finding"

    return concept_types, family_for_iri, top_level


def build_v2_json(parsed):
    iri_to_label = parsed["iri_to_label"]
    iri_to_label_en = parsed["iri_to_label_en"]
    parent_map = parsed["parent_map"]
    child_map = parsed["child_map"]
    restrictions = parsed["class_restrictions"]
    synonymes_map = parsed["class_synonymes"]
    annotations = parsed["class_annotations"]

    concept_types, family_for_iri, top_level = determine_types(parsed)
    all_iris = set(iri_to_label.keys())

    def iri_list_to_keys(iri_list):
        return [label_to_key(iri_to_label[i]) for i in iri_list if i in iri_to_label]

    def get_poids(iri):
        wt = restrictions.get(iri, {}).get("weight", [])
        if wt:
            wlabel = iri_to_label.get(wt[0], "")
            return POIDS_MAP.get(wlabel, 2)
        return 2

    def get_categorie(iri):
        family = family_for_iri.get(iri, "")
        if family == "Pathologie":
            return "DIAGNOSTIC_MAJEUR"
        if family == "Topographie":
            return "TOPOGRAPHIE"
        wt = restrictions.get(iri, {}).get("weight", [])
        if wt:
            wlabel = iri_to_label.get(wt[0], "")
            cat_map = {"Urgent": "DIAGNOSTIC_URGENT", "majeur": "DIAGNOSTIC_MAJEUR",
                       "moyen": "DIAGNOSTIC_MOYEN", "descriptif": "DESCRIPTION_ECG"}
            return cat_map.get(wlabel, "DESCRIPTION_ECG")
        ct = concept_types.get(iri, "finding")
        if ct == "pattern":
            return "DIAGNOSTIC_MAJEUR"
        if ct == "qualifier":
            return "QUALIFICATEUR"
        return "DESCRIPTION_ECG"

    concepts = {}
    skipped = 0
    for iri in sorted(all_iris, key=lambda x: iri_to_label.get(x, x)):
        ctype = concept_types.get(iri, "finding")
        if ctype == "reference":
            skipped += 1
            continue

        label = iri_to_label[iri]
        key = label_to_key(label)
        rels = restrictions.get(iri, {})
        annots = annotations.get(iri, {})

        c = {"concept_name": label}
        if iri in iri_to_label_en:
            c["concept_name_en"] = iri_to_label_en[iri]
        c["categorie"] = get_categorie(iri)
        c["poids"] = get_poids(iri)
        c["type"] = ctype

        for field in ["requires", "supports", "has_qualifiers", "has_qualifier_families",
                       "excludes", "excludes_families"]:
            vals = iri_list_to_keys(rels.get(field, []))
            if vals:
                c[field] = vals

        syns = synonymes_map.get(iri, [])
        if syns:
            c["synonymes"] = syns

        parents_in = [p for p in parent_map.get(iri, []) if p in all_iris]
        pkeys = iri_list_to_keys(parents_in)
        if pkeys:
            c["parents"] = pkeys
        children_in = [ch for ch in child_map.get(iri, []) if ch in all_iris]
        ckeys = iri_list_to_keys(children_in)
        if ckeys:
            c["children"] = ckeys

        for field, prop in [("territoires_possibles", "territory"),
                             ("origin_structure", "origin_structure"),
                             ("origin_territory", "origin_territory"),
                             ("electrode", "electrode"),
                             ("ecg_morphology", "ecg_morphology")]:
            vals = iri_list_to_keys(rels.get(prop, []))
            if vals:
                c[field] = vals

        if annots.get("hide"):
            c["hide"] = annots["hide"]
        if annots.get("mayhaveterritory"):
            c["mayhaveterritory"] = True
        if annots.get("importance_territoire"):
            c["importance_territoire"] = annots["importance_territoire"]
        if annots.get("acronym"):
            c.setdefault("synonymes", [])
            if annots["acronym"] not in c["synonymes"]:
                c["synonymes"].append(annots["acronym"])
        if annots.get("mayhavemirror"):
            c["mayhavemirror"] = annots["mayhavemirror"]

        concepts[key] = c

    # qualifier families
    qfamilies = {}
    for iri, rels in restrictions.items():
        for qfam_iri in rels.get("has_qualifier_families", []):
            fam_label = iri_to_label.get(qfam_iri)
            if fam_label:
                fam_key = label_to_key(fam_label)
                children = child_map.get(qfam_iri, [])
                if fam_key not in qfamilies:
                    qfamilies[fam_key] = {
                        "family_name": fam_label,
                        "members": iri_list_to_keys(children),
                    }

    # categories
    concept_categories = defaultdict(list)
    for key, c in concepts.items():
        concept_categories[c["categorie"]].append(key)

    # territoires
    territoires_ecg = {}
    for root_iri, root_label in top_level.items():
        if root_label == "Topographie":
            for child_iri in child_map.get(root_iri, []):
                cl = iri_to_label.get(child_iri, "")
                ck = label_to_key(cl)
                desc = get_all_descendants(child_iri, child_map)
                desc.discard(child_iri)
                territoires_ecg[ck] = {
                    "label": cl,
                    "sous_territoires": [iri_to_label.get(d, d) for d in desc],
                }

    meta = {
        "version": "2.0",
        "source": "BrYOzRZIu7jQTwmfcGsi35.owl",
        "generated_by": "convert_owl_to_v2.py",
        "total_concepts": len(concepts),
        "total_patterns": sum(1 for c in concepts.values() if c["type"] == "pattern"),
        "total_findings": sum(1 for c in concepts.values() if c["type"] == "finding"),
        "total_qualifiers": sum(1 for c in concepts.values() if c["type"] == "qualifier"),
        "total_topography": sum(1 for c in concepts.values() if c["type"] == "topography"),
        "total_qualifier_families": len(qfamilies),
        "total_synonymes": sum(len(c.get("synonymes", [])) for c in concepts.values()),
        "total_requires_relations": sum(len(c.get("requires", [])) for c in concepts.values()),
        "total_supports_relations": sum(len(c.get("supports", [])) for c in concepts.values()),
        "total_excludes_relations": sum(len(c.get("excludes", [])) for c in concepts.values()),
        "total_hidden": sum(1 for c in concepts.values() if c.get("hide")),
        "skipped_reference_concepts": skipped,
    }

    scoring_rules = {
        "poids_mapping": POIDS_MAP,
        "v2_scoring": {
            "pattern_base_score": 50,
            "requires_weight": 30,
            "qualifier_weight": 10,
            "supports_weight": 5,
            "exclusion_penalty": "zero_score",
        },
    }

    return {
        "concepts": concepts,
        "qualifier_families": qfamilies,
        "concept_categories": dict(concept_categories),
        "territoires_ecg": territoires_ecg,
        "scoring_rules": scoring_rules,
        "metadata": meta,
    }


if __name__ == "__main__":
    owl_path = Path(__file__).parent / "BrYOzRZIu7jQTwmfcGsi35.owl"
    output_path = Path(__file__).parent / "data" / "ontology_v2.json"
    print(f"[1/3] Parsing OWL: {owl_path.name}")
    parsed = parse_owl(owl_path)
    print(f"      {len(parsed['classes_raw'])} classes")
    print(f"[2/3] Building V2 JSON...")
    v2 = build_v2_json(parsed)
    for k, v in v2["metadata"].items():
        print(f"      {k}: {v}")
    print(f"[3/3] Writing: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(v2, f, ensure_ascii=False, indent=2)
    print(f"      {output_path.stat().st_size / 1024:.1f} KB written. Done!")
