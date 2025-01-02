# CHU Amélie
# BOGUSH Ekaterina

from collections import deque
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
from scipy.stats import chi2_contingency
from IPython.display import display
from scipy.cluster.hierarchy import DisjointSet
from typing import Set

""" 
----------- README -----------
Tout au long du projet on fait l'hypothèse que les valeurs de target possibles (classe d'un individu) sont 0 et 1.
Les réponses aux questions se trouvent à la fin du document.

Dans la partie 7:
Pour le classifieur TAN on a utilisé le lissage de Laplace dans estimProba pour éviter le problème des probabilités nulles. 
Par exemple, si la base de données contient les colonnes :
(A1 | A2 | Target) et que l'on veut estimer la probabilité (par la fréquence) de A1 = 1 sous la condition sur Target, on utilise la formule suivante :

P(A1 = 1 | Target = t) = (N1 + eps) / (N + Na * eps) avec
    - N : nombre de lignes telles que P(target = t)
    - N1 : nombre de lignes telles que P(target = t et A1 = 1)
    - Na : nombre de valeurs distinctes de la colonne A1 
"""

# --------
# Partie 1. Classification a priori
# --------


def getPrior(df: pd.DataFrame) -> dict:
    """
    Calcule la probabilité a priori de la classe 1 (target = 1) et l'intervalle de confiance à 95%.

    Parameters
    ---------
        df: pd.DataFrame 
            La base de données pour calculer l'a priori

    Returns
    -------
        Dict[str, float] : 
            Un dictionnaire contenant la probabilité a priori de la classe 1, borne inférieure et supérieure de l'intervalle de confiance.
    """

    # Probabilite est estimée par la fréquence

    # nombre d'individus
    n = df["target"].count()

    # nombre d'individus de classe target 1 : correspond à notre moyenne empirique
    proba = df['target'][df["target"] == 1].count() / n

    # estimation de l'ecart-type par la loi de Bernoulli
    std = np.sqrt(proba * (1 - proba))

    # Intervalle de confiance à 95%
    min5percent = -1.96 * std / np.sqrt(n) + proba  # moyenne = proba (estimation par Bernoulli)
    max5percent = 1.96 * std / np.sqrt(n) + proba

    return {'estimation': proba.item(), 'min5pourcent': min5percent.item(), 'max5pourcent': max5percent.item()}


class APrioriClassifier(utils.AbstractClassifier):
    """
    Un classifieur qui estime la classe d'un individu par a priori (la classe majoritaire)
    de la base de données à l'initialisation. Il propose aussi de calculer des statistiques
    de qualité à paritr d'un pandas.DataFrame.
    """

    def __init__(self, train_data: pd.DataFrame):
        """
        Initialise le classifieur en calculant la probabilité P(target = 1) par a priori sur 'train_data'.

        Parameters
        ----------
            train_data: pd.DataFrame
                La base de données sur laquelle un classifieur sera entraîné.
        """

        self.data = train_data
        self.prior = getPrior(train_data)
    
    def estimClass(self, d: dict = None) -> int:
        """
        Estime la classe de l'individu par a priori.

        Parameters
        ----------
            d: Dict[attr, int]
                Un dictionnaire contenant l'ensemble des attributs d'un individu, sous la forme 
                {'nom_attr' : valeur, ...}. Par défaut à None

        Returns
        -------
            int : la classe (target) d'un individu estimée.
        """

        return int(self.prior["estimation"] > 0.5)

    def statsOnDF(self, df: pd.DataFrame) -> dict:
        """
        Calcule des statistiques de qualité de l'estimateur sur la base de données 'df'. Les statistiques calculées:
            - VP: vrai positif
            - VN: vrai négatif
            - FP: faux positif
            - FN: faux négatif
            - précision = VP / (FN + VP)
            - rappel = VP / (VP + FP)

        Parameters
        ----------
            df: pd.DataFrame
                La base de données sur laquelle la qualité du classifieur est évaluée

        Returns
        -------
            Dict[str, float] : Les statistiques dans un dictionnaire de clés {VP, VN, FP, FN, precision, rappel}
            avec leurs valeurs correspondantes.
        """

        # initialisation
        vp, vn, fp, fn = 0, 0, 0, 0

        # parcours de la base de données df
        for t in df.itertuples(index=False):

            # on convertit le tuple en dictionnaire
            d = t._asdict()

            # classe estimée de l'inividu
            estim_target = self.estimClass(d)

            # la vraie classe de l'individu
            target = d["target"]

            if target and estim_target:             # 1 1
                vp += 1
            if target and estim_target == 0:        # 1 0
                fn += 1
            if target == 0 and estim_target:        # 0 1
                fp += 1
            if target == 0 and estim_target == 0:   # 0 0
                vn += 1

        # calcul de la précision et du rappel en évitant la division par 0
        precision = vp / (vp + fp) if (vp + fp) > 0 else 0
        rappel = vp / (vp + fn) if (vp + fn) > 0 else 0

        return {'VP': vp, 'VN': vn, 'FP': fp, 'FN': fn, 'precision': precision, 'rappel': rappel}

# --------
# Partie 2. Classification probabiliste a 2D
# --------


def P2D_l(df: pd.DataFrame, attr: str, diff_attr: int = 0, eps: float = 0) -> dict:
    """
    Calcule la distribution de probabilité P(attr | target) dans df. Si les paramètres 'diff_attr' et 'eps' sont non nuls,
    alors applique le lissage de Laplace dans le calcul.

    Parameters
    ----------
        df : pd.DataFrame
            La base de données utilisée pour calculer les probabilités
        attr : str
            L'attribut (colonne de 'df') pour lequel les probabilités sont calculées.
        diff_attr : int
            Le nombre de valeurs différentes d'attribut 'attr' dans 'df'
        eps : float 
            La valeur pour le lissage de Laplace

    Returns
    -------
        Dict[int, Dict[int, float]] : La distribution de probabilité sous forme d'un dictionnaire structurés
        de manière suivante  {'valeur_de_target' : {'valeur_de_attr' : P(attr = a | target = t), ...}.
    """

    # on compte le nombre d'entrées pour chaque combinaison attr target possibles
    tuples = df[[attr, 'target']].value_counts().to_dict()

    # dictionnaire qui pour chaque valeur de target va associer le nombre d'entrées avec cette valeur
    total_target = {t: 0 for t in [0, 1]}

    # calcul du nombre total de chaque valeur de target
    for (_, t), count in tuples.items():
        total_target[t] += count

    # pour chaque valeur de t on associe un dictionnaire de clés les valeurs d'attr
    proba = {t: {a.item(): (1 / diff_attr) if diff_attr else 0 for a in df[attr].unique()} for t in [0, 1]}

    # calcul de la probabilité P(attr|target) pour chaque valeur d'attr et de target (avec ou sans lissage)
    for (a, t), count in tuples.items():
        proba[t][a] = (count + eps) / (total_target[t] + eps * diff_attr)
    return proba


def P2D_p(df: pd.DataFrame, attr: str) -> dict:
    """
    Calcule la distribution de probabilité P(target | attr) dans df.

    Parameters
    ----------
        df : pd.DataFrame
            La base de données utilisée pour calculer les probabilités

        attr: str
            L'attribut (colonne de 'df') pour lequel les probabilités sont calculées.

    Returns
    -------
        Dict[int, Dict[int, float]] : La distribution de probabilité sous forme d'un dictionnaire structurés
        de manière suivante  {'valeur_de_attr' : {'valeur_de_target' : P(target = t | attr = a), ...}.
    """

    # on compte le nombre d'entrées pour chaque combinaison attr target possibles
    tuples = df[[attr, 'target']].value_counts().to_dict()

    # dictionnaire qui pour chaque valeur d'attr va associer le nombre d'entrées avec cette valeur
    total_attr = {att.item(): 0 for att in df[attr].unique()}

    # calcule du nombre total de chaque valeur d'attr
    for (attr, t), count in tuples.items():
        total_attr[attr] += count

    # pour chaque valeur d'attr on associe un dictionnaire de clés les valeurs de target
    proba = {a: {t: 0 for t in [0, 1]} for a in total_attr.keys()}

    # calcule de la probabilité P(target|attr) pour chaque valeur d'attr et de target
    for (a, t), count in tuples.items():
        proba[a][t] = count / total_attr[a]

    return proba


class ML2DClassifier(APrioriClassifier):
    """
    Un classifieur qui estime la classe d'un individu par maximum de vraisemblance (ML).
    Il propose aussi de calculer des statistiques de qualité de l'estimateur à partir d'un pandas.dataframe.

    Estimation du ML: estimation de la classe (target) d'un individu selon la probabilité P(attr|target).
    La classe estimée est celle ayant la probabilité (vraisemblance) la plus élevée par rapport aux autres classes (target).
    """

    def __init__(self, train_data: pd.DataFrame, attr: str):
        """
        Initialise le classifieur ML en l'entraînant sur les données 'train_data'.
        Un dictionnaire de probabilité initialisée contient les probabilités suivantes:
            - P(attr = a|target = t)
                (a : toutes les valeurs possibles de attr, t : toutes les valeurs possibles de target)

        Parameters :
        ---------- 
            train_data: 
                Les données (dataframe) sur lesquelles le classifieur sera entraîné.
            attr: str
                L'attribut sur lequel se base l'estimation de la classe (target)
        """

        super().__init__(train_data)
        self.P2Dl = P2D_l(train_data, attr)
        self.attr = attr

    def estimClass(self, d=None) -> int:
        """
        Estime la classe de l'individu représenté par le dictionnaire par maximum de vraisemblance

        Parameters
        ----------
            d: Dict[attr, valeur]
                Le dictionnaire d'attributs et leurs valeurs d'un individu 

        Returns
        -------
            int: la classe (target) estimée de l'individu
        """

        indiv_attr = d[self.attr]
        return int(self.P2Dl[1][indiv_attr] > self.P2Dl[0][indiv_attr])


class MAP2DClassifier(APrioriClassifier):
    """
    Un classifieur qui permet d'estimer la classe (target) d'un individu en fonction du maximum a posteriori (MAP).
    Il propose aussi de calculer des statistiques de qualité de l'estimateur à partir d'un pandas.dataframe.

    Estimation du MAP: estimation de la classe (target) d'un individu selon la probabilité P(target|attr).
    La classe estimée est celle ayant la probabilité la plus élevée par rapport aux autres classes (targets).
    """

    def __init__(self, train_data: pd.DataFrame, attr: str):
        """
        Initialise le classifieur MAP en l'entraînant sur les données 'train_data'.
        Un dictionnaire de probabilité initialisé contient les probabilités suivantes:
            - P(target = t|attr = a)
                (a : toutes les valeurs possibles de attr, t : toutes les valeurs possibles de target)

        Parameters :
        ---------- 
            train_data: 
                Les données (dataframe) sur lesquelles le classifieur sera entraîné.
            attr: str
                L'attribut sur lequel se base l'estimation de la classe (target)
        """

        super().__init__(train_data)
        self.P2Dp = P2D_p(train_data, attr)
        self.attr = attr

    def estimClass(self, d) -> int:
        """
        Estime la classe de l'individu représenté par le dictionnaire par maximum a posteriori

        Parameters
        ----------
            d: Dict[attr, valeur]
                Le dictionnaire d'attributs et leurs valeurs d'un individu 

        Returns
        -------
            int: La classe (target) estimée de l'individu 
        """

        indiv_attr = d[self.attr]
        return int(self.P2Dp[indiv_attr][1] > self.P2Dp[indiv_attr][0])

# --------
# Partie 3. Complexites en memoire
# --------


def convert(n: int) -> None:
    """
    Affiche la conversion du nombre d'octets 'n' en Go, Mo, Ko et octets.  
    L'affichage commence par la plus grande unité qui n'est pas égale à 0.  
    Par exemple, pour 5000 octets : 0 go, 0 mo, 4 ko, 100 o, mais uniquement "4 ko 100 o" sera affiché.

    Parameters:
        n : int > 0
            Le nombre d'octets dont la converstion il faut afficher
    """

    # Rien a convertir, i.e. n < ko
    if n < 1024:
        print('')
        return
    print("=", end=' ')

    unit = 1 << 30  # point de depart = 2^30 = go en octets
    names = ['o', 'ko', 'mo', 'go']
    print_flag = False  # flag du debut d'affichage

    i = 3
    while i >= 0:
        if unit == 0:  # unite = octets
            unit = 1
        q = 0
        if n >= unit:
            print_flag = True if unit != 1 else print_flag
            q = n // unit  # div euclidienne
            n -= q * unit

        if print_flag:
            print(f"{q}{names[i]}", end=' ')

        i -= 1  # passer a l'unite plus petite
        unit >>= 10  # bit shift droit = div par 2^10
    print('')


def nbParams(data: pd.DataFrame, l_attr: list[str] = None) -> int:
    """
    Calcule et affiche la taille de table de probabilité P(target | attr1, ..., attrN).
    La fonction ne compte que les probabilités (float) stockées.
    Toutes les combinaisons de possibles de valeurs des attributes doivent etre stockées.
    La table peut etre représenté de manière suivante:

                    | target = 0 | target = 1 |
    (a1, ..., an)   | proba      | proba      |
    (a1', ..., an') | proba      | proba      |

    Si on note N(attrI) : le nombre de valeurs distinctes de l'attribut I, alors la formule de la taille
    (en nombre de float) est suivante: N(target) * N(attr1) * ... * N(attr N).

    Hypothèse : un float est représenté sur 8 octets.

    Parameters
    ----------
        data : pd.DataFrame
            Les données (dataframe) pour lesquelles il faut calculer la table de probabilité.
        l_attr : 
            Liste des attributs dans 'data'. Si None, alors tous les attributs de 'data' sont considérés.

    Returns
    -------
        int : Le nombre d'octets nécessaire pour stocker la table de probabalité
    """

    # considérer tous les attributs
    if l_attr is None:
        l_attr = data.columns.values
    float_bytes = 8

    # N(attr1) * ... * N(attrN).
    all_bytes = data[l_attr].nunique().prod() * float_bytes

    # affichage
    print(f"{len(l_attr)} variable(s) : {all_bytes} octets", end=' ')
    convert(all_bytes)
    return all_bytes.item()


def nbParamsIndep(data: pd.DataFrame) -> int:
    """
    Calcule et affiche la taille de table de probabilité P(target, attr1, ..., attrN) en supposant
    que target et tous les attributs sont indépendants l'un de l'autre.
    La fonction ne compte que les probabilités (float) stockées. Il suffit donc de stocker
    P(attr1), ... pour toutes ces valeurs possibles -> les tableaux 1D sont suffisants.

    Si on note N(attrI) : le nombre de valeurs distinctes de l'attribut I, alors la formule de la taille
    (en nombre de float) est suivante: N(attr1) + ... + N(attrN).

    Hypothèse : un float est représenté sur 8 octets.

    Parameters
    ----------
        data : pd.DataFrame
            Les données (dataframe) pour lesquelles il faut calculer la table de probabilité.

    Returns
    -------
        int : Le nombre d'octets nécessaire pour stocker la table de probabalité.
    """

    float_bytes = 8
    # N(attr1) + ... + N(attrN)
    all_bytes = data.nunique().sum() * float_bytes

    # Affichage
    print(f"{len(data.columns)} variable(s) : {all_bytes} octets", end=' ')
    convert(all_bytes)
    return all_bytes.item()

# --------
# Partie 4. Classifiers Naïve Bayes
# --------


"""
Hypothèse du Naïve Bayes : P(attr1, ..., attrN | target) = P(attr1 | target) * ... * P(attrN | target)
[condition d'indépendance conditionnelle des attributs deux à deux conditionnellement à target].
"""


def drawNaiveBayes(df: pd.DataFrame, parent: str) -> None:
    """
    Dessine le graphe d'indépendance conditionnelle sous 'parent'.

    Parameters
    ----------
        df : pd.DataFrame
            Les données (dataframe) dont les attributs (colonnes) seront utilisés pour le graphe.
        parent : L'attribut (présent dans 'df') par rapport auquel le graphe sera construit i.e. P(attributs | parent).
    """

    # récupère tous les attributs
    l_attr = df.columns.values
    liste = []
    for attr in l_attr:
        if attr != parent:
            liste.append(f"{parent}->{attr}")
    chaine = ";".join(liste)
    return utils.drawGraph(chaine)


def nbParamsNaiveBayes(data: pd.DataFrame, parent: str, l_attr: list[str] = None) -> int:
    """
    Calcule et affiche la taille de table de probabilité en utilisant l'hypothèse du Naïve Bayes
    pour pouvoir calculer 
        - P(attr1, ..., attrN | target)
        - P(target | attr1, ..., attrN)

    Pour cela, il suffit de stocker P(attr1 | target), ..., P(attrN | target) ainsi que P(target).

    Si on note N(attrI) : le nombre de valeurs distinctes de l'attribut I, alors la formule de la taille
    (en nombre de float) est suivante: [N(attr1) + ... + N(attr N)] * N(target) + N(target).

    La fonction ne compte que les probabilités (float) stockées.
    Hypothèse : float est représenté sur 8 octets.

    Parameters
    ----------
        data : pd.DataFrame
            Les données (dataframe) pour lesquelles il faut calculer la table de probabilité.
        l_attr : Liste des attributs dans 'data' (['target', attr1, ...]).
            Si None, alors tous les attributs de 'data' sont considerés.

    Returns
    -------
        int : Le nombre d'octets nécessaire pour stocker la table de probabalité.
    """

    float_bytes = 8
    parent_nb = data[parent].nunique()  # nombre de valeurs distinctes de parent

    if l_attr is None:
        l_attr = data.columns
    else:
        l_attr = np.array(l_attr)

    # nombre d'attributs (y compris target)
    nb_vars = l_attr.size

    if nb_vars > 0:
        l_attr = l_attr[l_attr != parent]  # supprimer parent de la liste

    # on compte le nombre de valeurs uniques pour chaque attributs
    data_count = data[l_attr].nunique() if nb_vars > 0 else [0]

    all_bytes = (sum(data_count) + 1) * float_bytes * parent_nb  # +1 vient de la factorisation

    # affichage
    print(f"{nb_vars} variable(s) : {all_bytes} octets", end=' ')
    convert(all_bytes)
    return all_bytes


def init_mat_P2D(data_train: pd.DataFrame, prior_1: dict, eps=0) -> dict:
    """
    Initialise une table de probabilités sous forme de dictionnaire contenant les probabilités suivantes:
        - P(attr = a| target = t) pour chaque attribut attr colonne de df
        - P(target = t) 
            (a : toutes les valeurs possibles de attr, t : toutes les valeurs possibles de target)
    Le lissage de Laplace est appliqué uniquement si nécessaire (eps > 0).

    Parameters
    ----------
        data_train : pd.DataFrame
            Les données sur lesquelles la table sera calculée.
        prior_1 : Dict[str,float]
            Le dictionnaire contenant l'estimation de la classe 1 (target = 1) par a priori, et les intervalles de confiance à 95%
        eps : La valeur pour le lissage de Laplace. Si eps = 0, alors ne pas faire le lissage de Laplace.

    Returns
    -------
        Dict[str, dict]: la table de probabilités
            Le dictionnaire est sous la forme :
            {   attr      : { t : {P(attr = a| target = t)}}    pour chaque attribut attr != target colonne de data_train
                                                                pour a et t les valeurs possibles de attr et target
                'target'  : { 0 : P(target = 0), 1 : P(target = 1) }
            }

    """
    P = dict()

    for attr in data_train.columns.values:
        if attr != 'target':
            N_attr = 0
            if eps:
                N_attr = data_train[attr].nunique()  # nombre de valeurs différentes de l'attribut
            P[attr] = P2D_l(data_train, attr, N_attr, eps)

    P['target'] = {0: 1-prior_1['estimation'], 1: prior_1['estimation']}
    return P


class MLNaiveBayesClassifier(APrioriClassifier):
    """
    Un classifieur basé sur l'hypothèse de Naïve Bayes et qui permet d'estimer la classe (target) d'un individu
    en fonction du maximum vraisemblance (ML).

    Estimation du ML: estimation de la classe (target) d'un individu selon la probabilité P(attr1, ..., attrN | target).
    La classe estimée est celle ayant la probabilité (vraisemblance) la plus élevée par rapport aux autres classes (targets).
    """

    def __init__(self, data_train: pd.DataFrame):
        """
        Initialise le classifieur ML en l'entraînant sur les données 'train_data'.
        Une table de probabilité initialisée contient les probabilités suivantes:
            - P(attr = a| target = t) pour chaque attribut attr 
            - P(target = t) 
                (a : toutes les valeurs possibles de attr, t : toutes les valeurs possibles de target)

        Parameters :
        ---------- 
            train_data: Les données (dataframe) sur lesquelles le classifieur sera entraîné.
        """
        super().__init__(data_train)
        self.P2D_l_ml = init_mat_P2D(data_train, self.prior)

    def estimProbas(self, d: dict) -> dict:
        """
        Calcule la vraissemblance [P(attr1, ..., attrN | target)] pour chaque valeur de target pour un individu

        Parameters
        ----------
            d: Dict[str,int]
                Un dictionnaire contenant l'ensemble des attributs d'un individu, sous la forme 
                {'nom_attr' : valeur, ...}.

        Returns
        -------
            Dict[int, float] : la vraissemblance pour chaque valeurs de target.
            Le dictionnaire est sous la forme {0 : vraissemblance pour target = 0, 1 : vraissemblance pour target = 1}
        """

        # probabilites
        targets = {0, 1}
        probas = {t: 1 for t in targets}  # target : P(a1 ... an | target)
        for t in targets:
            # parcours des attributs et leurs valeurs de l'individu
            for attr, val in d.items():

                # verification de la presence de l'attribut dans P2D_l_mat
                if attr in self.P2D_l_ml and attr != 'target':  # on ne calcule pas target

                    if val in self.P2D_l_ml[attr][t]:
                        # multiplication par P(attr|target = t)
                        probas[t] *= self.P2D_l_ml[attr][t][val]
                    else:
                        # val pas dans P2D_l_mat[attr] alors P(attr|target = 0) = 0
                        probas[t] = 0
                        break

        return probas

    def estimClass(self, d: dict) -> int:
        """
        Estime la classe de l'individu suivant maximum de vraissemblance sous l'hypothèse du Naïve Bayes

        Parameter
        ---------
        d: Dict[str, int]
            Un dictionnaire contenant l'ensemble des attributs d'un individu, sous la forme 
                {'nom_attr' : valeur, ...}.

        Returns
        -------
            int : la classe (target) d'un individu estimée.
        """

        prob = self.estimProbas(d)
        return int(prob[1] > prob[0])


class MAPNaiveBayesClassifier(APrioriClassifier):
    """
    Un classifieur base sur l'hypothèse de Naïve Bayes et qui permet d'estimer la classe (target) d'un individu
    en fonction du maximum a posteriori (MAP).

    Estimation du MAP: estimation de la classe (target) d'un individu selon la probabilité P(target | attr1, ..., attrN).
    La classe estimée est celle ayant la probabilité la plus élevée par rapport aux autres classes (targets).
    """

    def __init__(self, train_data: pd.DataFrame):
        """
        Initialise le classifieur MAP en l'entraînant sur les données 'train_data'.
        Une table de probabilité initialisée contient les probabilités suivantes:
            - P(attr = a| target = t) pour chaque attribut attr 
            - P(target = t) 
                (a : toutes les valeurs possibles de attr, t : toutes les valeurs possibles de target)

        Parameters :
        ---------- 
            train_data: Les données (dataframe) sur lesquelles le classifieur sera entraîné.
        """

        super().__init__(train_data)
        self.P2D_map = init_mat_P2D(train_data, self.prior)

    def estimProbas(self, indiv_data: dict) -> dict:
        """
        Calcule les probabilités P(target | attr1, ..., attrN) de chaque valeur de target pour un individu.

        Parameters
        ----------
            indiv_data : Dict[str, int]
                Un dictionnaire contenant l'ensemble des attributs d'un individu, sous la forme 
                {'nom_attr' : valeur, ...}.

        Returns
        -------
            Dict[int, float] : un dictionnaire contenant les probabilités posteriori pour l'individu d'être à chaque classe. 
            Le dictionnaire est sous la forme {0 : probabilité que target = 0, 1 : probabilité que target = 1}
        """

        targets = {0, 1}
        probas = {target: 1 for target in targets}
        for target in targets:
            # Calc P(attr1 | target) * ... * P(attrN | target)
            for attr, val in indiv_data.items():
                if attr != 'target' and attr in self.P2D_map:
                    if val in self.P2D_map[attr][target]:
                        probas[target] = probas[target] * self.P2D_map[attr][target][val]
                    else:
                        probas[target] = 0
                        break
            # P(attr1 | target) * ... * P(attrN | target) * P(target)
            probas[target] *= self.P2D_map['target'][target]
        s = sum(probas.values())

        # Trouver proba P(target | attr1, ..., attrN)
        for target in targets:
            probas[target] = probas[target] / s if s else 0

        return probas

    def estimClass(self, indiv_data: dict) -> int:
        """
        Estime la classe d'un individu selon le principe du MAP.

        Parameters
        ----------
            indiv_data : Dict[str, int]
                Un dictionnaire contenant l'ensemble des attributs d'un individu, sous la forme 
                {'nom_attr' : valeur, ...}.

        Returns
        -------
            int : la classe (target) d'un individu estimée.
        """

        estim_proba = self.estimProbas(indiv_data)
        return int(estim_proba[1] > estim_proba[0])

# --------
# Partie 5. Les classifiers Naive Bayes reduits
# --------


def isIndepFromTarget(df: pd.DataFrame, attr: str, crit_level: float) -> bool:
    """
    Vérifie si l'attribut est indépendant du target selon les données (dataframe 'df'). 
    La vérification est effectuée à l'aide du test du khi-deux avec l'hypothèse H0 posée : attribut est indépendant du target.

    Parameters
    ----------
        df : pd.DataFrame
            Les données (dataframe) à partir desquelles il faut vérifier si l'attribut est indépendant de target.
        attr : str
            Attribut présent dans 'df' l'indépendence duquel sera teste.
        crit_level : float
            Le niveau de signification (la valeur à partir de laquelle l'hypothèse H0 est acceptée)
    Returns
    -------
        bool : True si H0 est acceptée (attribut est indépendant du target).
               False si H0 est rejetée (attribut influence target).
    """

    # calcul du tableau de contingence
    contingency_table = pd.crosstab(index=df[attr], columns=df['target'])

    # test du khi-deux avec H0 : attr est independant du target
    res = chi2_contingency(contingency_table)

    # Verifier si pvalue est dans la zone de rejet
    # si p < crit_level -> rejet
    # si p >= crit_level -> accept
    return res.pvalue >= crit_level


def getDepFromTarget(df: pd.DataFrame, crit_level: float) -> list[str]:
    """
    Trouve toutes les attributs qui influencent target (le test d'indépendance est faite avec le test du khi-deux,
    voir la fonction 'isIndepFromTarget' pour plus de détails).

    Parameters
    ----------
        df : pd.DataFrame
        crit_level : float
            Le niveau de signification (la valeur à partir de laquelle l'hypothèse H0 est acceptée)

    Returns
    -------
        list[str] : Liste des attributs qui ne sont pas indépendants du target.
    """

    attrs = [attr for attr in df.columns.values if attr != 'target' and (not isIndepFromTarget(df, attr, crit_level))]
    attrs.append('target')
    return attrs


class ReducedMLNaiveBayesClassifier(MLNaiveBayesClassifier):
    """
    Un classifieur basé sur l'hypothèse de Naïve Bayes et qui permet d'estimer la classe (target) d'un individu
    en fonction du maximum vraisemblance (ML). Les attributs qui sont pris en compte pour faire une estimation 
    sont ceux qui influencent target (i.e. target et l'attribut ne sont pas indépendants). La vérification 
    de l'indépendance entre target et chaque attribut est effectuée a l'aide du test de khi-deux en posant 
    H0 : attribut et target sont indépendants.

    Estimation du ML: estimation de la classe (target) d'un individu selon la probabilité P(attr1, ..., attrN | target).
    La classe estimée est celle ayant la probabilité la plus élevée par rapport aux autres classes (target).
    """

    def __init__(self, train_data: pd.DataFrame, crit_level: float):
        """
        Initialise le classifieur ML en le entraînnant sur les données 'train_data'.
        Une table de probabilité initialisée contient les probabilités suivantes:
            - P(attr = a| target = t) pour chaque attribut attr 
            - P(target = t) 
                (a : toutes les valeurs possibles de attr, t : toutes les valeurs possibles de target)

        Parameters :
        ---------- 
            train_data: Les données (dataframe) sur lesquelles le classifieur sera entraîné.
            crit_level: Le niveau de signification pour le test de khi-deux (vérification de l'indépendence entre target
                et de chaque attribut).
        """

        super().__init__(train_data.filter(getDepFromTarget(train_data, crit_level), axis='columns'))

    def draw(self) -> None:
        """
        Dessine le graphe d'indépendence conditionnelle des attributs les uns par rapport aux autres conditionnellement à target
        (grace a l'hypothèse de Naïve Bayes).
        """

        attrs = self.P2D_l_ml.keys()
        edges = ';'.join([f'target->{attr}' for attr in attrs if attr != 'target'])
        display(utils.drawGraph(edges))


class ReducedMAPNaiveBayesClassifier(MAPNaiveBayesClassifier):
    """
    Un classifieur basé sur l'hypothèse de Naïve Bayes et qui permet d'estimer la classe (target) d'un individu
    en fonction du maximum à posteriori (MAP). Les attributs qui sont pris en compte pour faire une estimation 
    sont ceux qui influencent le target (i.e. target et l'attribut ne sont pas indépendants). La vérification 
    de l'indépendance entre target et chaque attribut est effectuée à l'aide du test de khi-deux avec 
    H0 : attribut et target sont indépendants.

    Estimation du MAP: estimation de la classe (target) d'un individu selon la probabilité P(target | attr1, ..., attrN).
    La classe estimée est celle ayant la probabilité la plus élevée par rapport aux autres classes (targets).
    """

    def __init__(self, train_data: pd.DataFrame, crit_level: float):
        """
        Initialise le classifieur MAP en le entraînnant sur les données 'train_data'.
        Une table de probabilité initialisée contient les probabilités suivantes:
            - P(attr = a| target = t) pour chaque attribut attr 
            - P(target = t) 
                (a : toutes les valeurs possibles de attr, t : toutes les valeurs possibles de target)

        Parameters :
        ---------- 
            train_data: Les données (dataframe) sur lesquelles le classifieur sera entraîné.
            crit_level: Le niveau de signification pour le test de khi-deux (verification de l'indépendence entre target
                et de chaque attribut).
        """

        super().__init__(train_data.filter(getDepFromTarget(train_data, crit_level), axis='columns'))

    def draw(self):
        """
        Dessine le graphe d'indépendence conditionnelle des attributs les uns par rapport aux autres sous target
        (grace a l'hypothèse de Naïve Bayes).
        """

        attrs = self.P2D_map.keys()
        edges = ';'.join([f'target->{attr}' for attr in attrs if attr != 'target'])
        display(utils.drawGraph(edges))
        

# --------
# Partie 6. Évaluation des classifieurs
# --------

def mapClassifiers(dic: dict, df: pd.DataFrame) -> None:
    """
    Représente graphiquement les classifieurs avec en coordonnées (x, y) = (précision, rappel)

    Parameters
    ----------
        dic: Dict[nom, instance de classifieur] 
            le dictionnaire des noms et des instances de classifieurs à représenter.
        df: DataFrame
            la base de données sur lequel on va évaluer la précision et le rappel
    """

    # objet figure
    fig = plt.figure(facecolor='lightgray')

    # ajout d'un seul sous-graphe dans la figure
    ax = fig.add_subplot(111)

    # parcours des instances de classifieurs
    for name, instance in dic.items():

        # calcul des statistiques de qualité
        stats = instance.statsOnDF(df)
        x, y = stats['precision'], stats['rappel']

        # tracer le point
        ax.plot(x, y, 'rx')
        # ajout du nom du point
        ax.text(x, y, name, fontsize=10, verticalalignment='bottom', horizontalalignment='left')

    # Ecriture du nom des axes
    ax.set_xlabel('précision')
    ax.set_ylabel('rappel')

    # Affichage de la figure
    plt.show()

# --------
# Partie 7. Sophistication du modèle
# --------

def MutualInformation(df: pd.DataFrame, x: str, y: str) -> float:
    """
    Calcule I(X, Y) l'information mutuelle des variables x et y. Plus I(X, Y) est élevé, plus il y a un forte dépendance
    entre X et Y.

    Parameters
    ----------
        df: pandas.DataFrame
            la base de données contenant les variables x, y et z
        x: str
            la première variable à évaluer
        y: str
            la deuxième variable à évaluer

    Returns
    -------
        float:
            l'indépendance mutuelle des variables x et y    
    """

    # estimation des probas par les fréquences
    joint_proba = df.groupby([x, y]).size() / len(df.index)  # P(X, Y)
    x_proba = df[x].value_counts(normalize=True)  # P(X)
    y_proba = df[y].value_counts(normalize=True)  # P(Y)

    # variable I(X, Y)
    I = 0  
    for (a1, a2) in joint_proba.index:
        joint_p = joint_proba[(a1, a2)]
        # estimation par la formule donnée
        I += joint_p * np.log2(joint_p / (x_proba[a1] * y_proba[a2]))
    return I


def ConditionalMutualInformation(df: pd.DataFrame, x: str, y: str, z: str) -> float:
    """
    Calcule I(X, Y|Z) l'information mutuelle des variables x et y conditionnellement à la variable z (target).
    Plus I(X, Y|Z) est élevé, plus il y a une forte dépendance entre X et Y conditionnées à Z.

    Parameters
    ----------
        df: pandas.DataFrame
            la base de données contenant les variables x, y et z
        x: str
            la première variable à évaluer
        y: str
            la deuxième variable à évaluer
        z: str
            variable de conditionnement  

    Returns
    -------
        float:
            l'indépendance mutuelle des variables x et y conditionnées à z
    """

    # cas même variable
    if x == y:
        return 0

    # calcul des probabilités
    joint_proba = df.groupby([x, y, z]).size() / len(df.index)  # P(X, Y, Z)
    xz_proba = df.groupby([x, z]).size() / len(df.index)        # P(X, Z)
    yz_proba = df.groupby([y, z]).size() / len(df.index)        # P(Y, Z)
    z_proba = df[z].value_counts(normalize=True)                # P(Z)

    # variable I(X,Y|Z)
    I = 0

    # parcours sur les valeurs de x, z, y dans df
    for (x1, y1, z1) in joint_proba.index:
        # numérateur et dénominateur de la division
        num = z_proba[z1] * joint_proba[(x1, y1, z1)]
        denom = xz_proba[(x1, z1)] * yz_proba[(y1, z1)]

        # estimation par la formule donnée
        I += joint_proba[(x1, y1, z1)] * np.log2(num/denom)

    return I


def MeanForSymetricWeights(matrix: np.array) -> float:
    """
    Calcule la moyenne des valeurs dans la matrice 'matrix' symétrique et de diagonale nulle.

    Parameters
    ----------
        matrix : np.array
            Matrice symétrique et de diagonale nulle.
    Returns
    -------
        La moyenne de la matrice.
    """
    
    # matrix est de la taille dim * dim
    dim, _ = matrix.shape
    # la somme des termes de la matrice de diagonale nulle est la somme des termes du triangle supérieur x2
    S = np.sum(matrix[np.triu_indices(n=dim, k=1)]) * 2

    # on ne prend pas en compte la diagonale dans le calcul du poids
    return S / (dim * dim - dim)  


def SimplifyConditionalMutualInformationMatrix(matrix: np.array) -> None:
    """
    Modifie la matrice 'matrix' sur place en mettant à 0 les valeurs qui sont plus petites que
    la moyenne des valeurs de la matrice. Les valeurs qui sont plus grandes que la moyenne ne sont pas modifiées.

    Parameters
    ----------
        matrix : np.array
            Matrice symétrique et de diagonale nulle.
    """

    mean = MeanForSymetricWeights(matrix)
    matrix[matrix < mean] = 0


def Kruskal(data: pd.DataFrame, matrix: np.array) -> list[tuple[str, str, float]]:
    """
    Construit l'arbre (ou une forêt) couvrant de poids maximal. L'algorithme de Kruskal est utilisé.
    Le graphe (non orienté) est construit de manière suivante:
        - Les sommets : tous les attributs de data (sauf target)
        - Les arêtes : une arête (s1, s2) existe ssi I(s1, s2 | target) > 0, soit matrix[s1][s2] > 0.

    Parameters
    ----------
        data : pd.DataFrame
            La base de données utilisée.

        matrix : 
            La matrice réduite des informations entre attributs conditionnellement à target. L'association des lignes et
            des colonnes aux attributs respecte leur ordre dans data. Par exemple, si age est la 1ère colonne de data,
            alors la ligne 0 et la colonne 0 représentent l'attribut age.
            Ainsi, si sex est la 2ème colonne de data, la case à l'indice (0, 1) contient I(age, sex | target).
            Par conséquent, la matrice est symétrique et de diagonale nulle.

            Réduction de la matrice : les valeurs inférieures à la moyenne des valeurs de la matrice sont à 0
            (elles sont considérées comme très peu dépendantes conditionnellement à target).

    Returns
    -------
        Les arcs du graphe choisit par l'algorithme de Kruskal.
    """

    dim = matrix.shape[0]

    # Sommets = tous les attributs
    V = [attr for attr in data.keys() if attr != 'target']

    # Aretes = (attr1, attr2, poids) si I(attr1, attr2 | target) > 0 (i.e. très dépendants)
    E = [(V[i], V[j], matrix[i][j]) for i in range(dim) for j in range(dim) if matrix[i][j] > 0]

    # liste d'arêtes choisies
    chosen_E = []
    union_find = DisjointSet(V)  # init Union-Find
    E.sort(key=lambda t: t[2], reverse=True)  # trier par ordre décroissant des poids

    for edge in E:
        attr1, attr2, _ = edge
        # Si la chaîne n'existe pas déjà parmi les arêtes choisies, ajout de l'arête
        if not union_find.connected(attr1, attr2):
            chosen_E.append(edge)
            union_find.merge(attr1, attr2)
    return chosen_E


def ConnexSets(edges: list[tuple[str, str, float]]) -> list[Set[str]]:
    """
    Trouve les composantes connexes dans le graphe non orienté dont les arêtes sont donné par 'edges'.
    Le parcours en largeur (BFS) est utilisé pour les retrouver.

    Parameters
    ----------
        edges : list[tuple[str, str, float]]
            Les arêtes d'extrémités les str, avec leurs poids qui existent dans le graphe.
    Returns
    -------
        list[Set[str]] : Une liste des composantes connexes (l'ensemble des sommets) dans le graphe.
    """

    # initialisation 
    E = dict()          # dictionnaire d'adjacence pour les sommets 
    visited = dict()    # dictionnaire de sommets visités ou non

    # parcours de tous les arêtes
    for v1, v2, _ in edges:
        # les sommets v1 et v2 sont adjacents {v1 : {v2}, v2 : {v1}}
        E.setdefault(v1, set()).add(v2)
        E.setdefault(v2, set()).add(v1)

        # les sommets v1 et v2 sont initialisé visité = False, {v1 : False, v2 : False}
        visited.setdefault(v1, False)
        visited.setdefault(v2, False)
    
    # liste de composantes connexes
    connected_sets = []

    # parcours des sommets
    for v in E.keys():
        # parcours en largeur de ce sommet
        if not visited[v]:
            connected_sets.append(BFS(E, visited, v))
    return connected_sets


def BFS(E: dict, visited: dict, root: str) -> Set[str]:
    """
    Le parcours en largeur utilisé pour retrouver les composantes connexes dans un graphe
    (fonction auxiliaire pour 'ConnexSets').

    Parameters
    ----------
        E : Dict[str, Set[str]] 
            Un dictionnaire représentant une liste d'adjacence du graphe. Le dictionnaire est 
            de la forme {'sommet': {sommets adjacents}}.
        visited : Dict[str, bool]
            Un dictionnaire qui indique pour chaque sommet s'il a été visité ou non 
        root : str
            Un sommet du graphe à partir duquel il faut commencer le parcours.

    Returns
    -------
        Set[str] : Un ensemble des sommets appartenant à la même composante connexe.
    """

    # Initialisation
    Q = deque()  # queue FIFO
    visited[root] = True  # les sommets visités
    explored = set()  # composante connexe
    explored.add(root)
    Q.append(root)

    # BFS
    while len(Q): # tant que la file n'est pas vide
        curr_v = Q.popleft()
        for v_n in E[curr_v]:  # sommets adjacents
            if visited[v_n] == False:
                visited[v_n] = True
                Q.append(v_n)
                explored.add(v_n)
    return explored


def OrientTree(root: str, V: Set[str], edges: list[tuple[str, str, float]], oriented_edges: list[tuple[str, str]]) -> None:
    """
    Oriente des arêtes du graphe non orienté qui contient une seule composante connexe à partir du sommet-racine 'root'.
    Les arcs sont ajoutées dans une liste 'oriented_edges' [modification sur place] (fonction auxiliaire pour 'OrientConnexSets').

    Parameters
    ----------
        root : str
            Un sommet-racine à partir duquel il faut orienter le graphe.
        V : Set[str]
            Un ensemble des sommets appartenant au graphe. Par conséquent, ils appartiennent à la même composante connexe.
        edges : list[tuple[str, str, float]]
            Les arêtes avec leurs poids qui existent dans le graphe.
        oriented_edges : list[tuple[str, str]]
            La liste des arcs orientés. (str1, str2) indique un arc str1->str2
    """

    visited = {v: False for v in V}  # les sommets visités
    E = dict()  # liste d'adjacence des sommets

    # initialise la liste d'adjacence de chaque sommets pour chaque arc.
    for v1, v2, _ in edges:
        E.setdefault(v1, set()).add((v2))
        E.setdefault(v2, set()).add((v1))

    
    # DFS parcours en profondeur
    stack = [root]

    # tant que la pile n'est pas vide
    while len(stack):
        curr_v = stack[-1]
        stack.pop()
        visited[curr_v] = True
        # les sommets adjacents
        for v in E[curr_v]:
            if not visited[v]:
                stack.append(v)
                oriented_edges.append((curr_v, v))


def OrientConnexSets(df: pd.DataFrame, edges: list[tuple[str, str, float]], classe: str) -> list[tuple[str, str]]:
    """
    Oriente un arbre (une forêt) couvrant donné par 'edges'. La racine est le sommet (l'attribut) qui maximise
    l'information mutuelle I(attr | classe = target). Voir la fonction 'Kruskal' pour plus d'information sur le graphe 
    à partir duquel l'arbre couvrant est retrouvé.

    Parameters
    ----------
        df: pd.DataFrame 
            La base de données utilisée.
        edges : list[tuple[str, str, float]]
            Une liste des arêtes d'un arbre (forêt) couvrant.
        classe : Un attribut (=target) conditionnellement à lequel il faut retrouver l'information mutuelle.

    Returns
    -------
        list[tuple[str, str]] : Une liste des arcs (arêtes d'un arbre (forêt) couvrant orientés à partir de la racine choisie).
    """

    # composantes connexes
    connected_sets = ConnexSets(edges)

    
    I = [] # liste des informations mutuelles
    oriented_edges = [] # liste d'arcs orientés

    for s in connected_sets:
        I = [(attr, MutualInformation(df, attr, classe)) for attr in s]
        root = max(I, key=lambda t: t[1])[0]
        OrientTree(root, s, edges, oriented_edges)
    return oriented_edges


class MAPTANClassifier(APrioriClassifier):
    """Classifieur basé sur Tree-augmented Naïve Bayes. Un attribut peut dépendre d'au plus 2 parent : target et un autre attribut
    si ces attributs ne sont pas indépendants l'un de l'autre.

    L'indépendance des attributs est vérifiée avec l'information mutuelle.

    La matrice des informations mutuelles entre attributs contionnellement à target, est consruite dont chaque
    ligne et colonne est associée à un attribut (sauf target). Par exemple, si l'attribut X est associé à
    la ligne X et l'attribut Y est associé à la colonne Y, alors la case [X, Y] contient
    l'information mutuelle I(X, Y | target).

    Les attributs sont considérés indépendants ssi leur information mutuelle est inférieure à la moyenne
    des valeurs dans la matrice.

    Lors d'une estimation de classe on utilise le lissage de Laplace.
    """

    def __init__(self, train_data: pd.DataFrame):
        """
        Initialise le classifieur en lui entraînant sur la base de données 'train_data'. Un graphe (un arbre orienté)
        est initialisé par une liste des sommets et une liste des prédecesseurs pour chaque sommet (au plus 2 parents). Une liste
        des arcs entre les attributs (target exclus) est stockée aussi.
        Le paramètre epsilon pour le lissage de Laplace est initialisé.

        Parameters
        ----------
            train_data : pd.DataFrame
                La base de données sur laquelle le classifieur sera entraînée.
        """

        super().__init__(train_data)

        # les sommets (attributs + target)
        self.graph_V = list(train_data.keys()) 

        # liste de prédécesseurpour chaque sommet et liste d'arcs pour entres les attributs
        self.graph_E_pred, self.attr_edges = self._create_edges(train_data, self.graph_V)

        # attributs ayant juste target en parent
        self.unique_parent_attrs = [k for k, pred in self.graph_E_pred.items() if len(pred) == 1]

        # attributs ayant 2 parents
        self.two_parents_attrs = [k for k, pred in self.graph_E_pred.items() if len(pred) == 2]

        # paramètres pour le lissage de Laplace
        self.eps = 1

        # init la matrice des probas P(attr | target) pour les attributs qui n'ont qu'un seul parent
        self.P2D_l = init_mat_P2D(train_data[self.unique_parent_attrs + ['target']], self.prior, self.eps)
        self.data = train_data

    def _proba_jointe_mult(self, parent: str, val_parent: int, attr: str) -> tuple[dict, dict, int]:
        """
        Calcule la probabilité P(attr = a | parent = val_parent, target = t (t dans {0, 1})) 
        dans le cas où l'attribut possède 2 parents.

        Parameters
        ----------
            parent:
                Le parent à conditionner (2ième parent qui n'est pas target)
            val_parent:
                La valeur du parent 
            attr:
                l'attribut conditionné à évaluer

        Returns
        -------
            - Dict[int, float] : Dictionnaire de probabilité (avec le lissage de Laplace appliqué) 
                                 pour chaque classe, soit {0 : P(attr = a | parent = val_parent, target = 0)
                                                           1 : P(attr = a | parent = val_parent, target = 1)}
                                 pour toutes valeurs a de attr présentes dans la base de données d'entraînement où parent = val_parent
                                 (base de données réduite).

            - Dict[Dict[int, int]] : Dictionnaire de nombre de lignes dans la base de données réduites pour chaque valeur de target, soit
                                     {0 : nombre de lignes tq parent = val_parent, target = 0,
                                      1 : nombre de lignes tq parent = val_parent, target = 1}

            - int : Le nombre de valeurs différentes d'attribut 'attr' dans la base de données d'entraînement (sans réduction)
        """

        # copie de la base de données
        tmp_data = self.data.copy()

        # réduire le dataset aux lignes parent = val_parent (pour P(attr | target, parent = val_parent))
        tmp_data = tmp_data[tmp_data[parent] == val_parent]

        targets = {0, 1}
        target_count = tmp_data['target'].value_counts()

        # nombre des valeurs de target = 0 et target = 1 dans le dataset réduit
        total = {t: 0 if t not in target_count.index else target_count[t] for t in targets}

        # nombre de valeurs différentes d'attributs (pour lissage)
        diff_attr = self.data[attr].nunique()
        return P2D_l(tmp_data, attr, diff_attr, self.eps), total, diff_attr

    def estimClass(self, indiv_data: dict) -> int:
        """
        Estime la classe d'un individu selon le principe du MAPTAN.

        Parameters
        ----------
            indiv_data : Dict[str, int]
                Un dictionnaire contenant l'ensemble des attributs d'un individu, sous la forme
                {'nom_attr' : valeur, ...}.

        Returns
        -------
            int : la classe (target) d'un individu estimée.
        """

        estim_proba = self.estimProbas(indiv_data)
        return int(estim_proba[1] > estim_proba[0])

    def estimProbas(self, indiv_data: dict) -> dict:
        """
        Calcule les probabilités P(target | attr1, ..., attrN) de chaque valeur de target pour un individu suivant
        le modèle TAN (Tree-augmented Naïve Bayes). Chaque attribut a au moins target comme parent et au plus un
        autre attribut comme parent.
        On note que P(attr1, ..., attrN | target) = P(target) * P(attrRacines | target) * Produit[P(attrI | ParentI, target)]

        Parameters
        ----------
            indiv_data : Dict[str, int]
                Un dictionnaire contenant l'ensemble des attributs d'un individu, sous la forme
                {'nom_attr' : valeur, ...}.

        Returns
        -------
            Dict[int, float] : un dictionnaire contenant les probabilités posteriori pour l'individu d'être à chaque classe.
            Le dictionnaire est sous la forme {0 : probabilité que target = 0, 1 : probabilité que target = 1}
        """

        targets = {0, 1}
        probas = {t: 1 for t in targets}

        # parcours de l'individu
        for attr, val in indiv_data.items():  
            if attr == 'target':
                continue
            # si l'attribut n'a qu'un seul parent
            if attr in self.unique_parent_attrs:
                for target in targets:
                    prob = self.P2D_l[attr][target][val]
                    probas[target] *= prob

            elif attr in self.two_parents_attrs:
                # Trouver le second parent (non target) de 'attr'
                parent = next(parent for parent in self.graph_E_pred[attr] if parent != 'target')
                # Calculer P(attr | target = t, parent = val_parent)
                proba_joint, total, diff_attr = self._proba_jointe_mult(parent, indiv_data[parent], attr)

                for target in targets:
                    # Appliquer le lissage de Laplace si la proba (estimée par la fréquence) = 0 pour éviter une probabilité nulle
                    if target not in proba_joint.keys():
                        probas[target] *= self.eps / (diff_attr * self.eps + total[target])  # lissage
                    elif val not in proba_joint[target]:
                        probas[target] *= self.eps / (diff_attr * self.eps + total[target])  # lissage
                    else:
                        probas[target] *= proba_joint[target][val]  # lissage déjà appliqué

        # multiplication par P(target)
        for target in targets:
            probas[target] *= self.prior['estimation'] if target == 1 else 1 - self.prior['estimation']

        # Proba totale
        s = sum(probas.values())

        # Trouver proba P(target | attr1, ..., attrN)
        for target in targets:
            probas[target] = probas[target] / s if s else 0

        return probas

    def _create_edges(self, train_data: pd.DataFrame, V: list[str]) -> tuple[dict, list]:
        """
        Initialise une liste des prédecesseurs pour chaque sommet et une liste des arcs entre les attributs (target exclus).
        Calcule la matrice réduite des informations mutuelles entre attributs contionnellement à target.

        Parameters:
            train_data: pd.DataFrame
                La base de données sur laquelle le classifieur sera entraînée.
            V : list[str]
                Une liste de tous les sommets (y compris target) dans un graphe.
        Returns
        -------
            Dict[str, Set[str]] : Un dictionnaire représentant une liste des prédecesseurs pour chaque sommet.
            List[tuple[str, str]] : Une liste des arcs entre les attributs (target exclus) dans un arbre.
        """

        # Matrice des informations mutuelles entre attributs contionnelement à target
        cmis = np.array([[0 if x == y else ConditionalMutualInformation(train_data, x, y, "target")
                        for x in train_data.keys() if x != "target"]
                         for y in train_data.keys() if y != "target"])

        # Réduction de la matrice
        SimplifyConditionalMutualInformationMatrix(cmis)

        # Les arcs orientés d'un arbre du Naïve Bayes. Target est exclus (pour fonction draw)
        attr_edges = OrientConnexSets(train_data, Kruskal(train_data, cmis), 'target')
        edges_pred = {attr: ['target'] for attr in V if attr != 'target'}  # target est le parent de chaque attribut
        edges_pred['target'] = []

        # trouver l'autre parent de chaque attribut
        for attr1, attr2 in attr_edges:
            edges_pred[attr2].append(attr1)

        return edges_pred, attr_edges

    def draw(self):
        """Dessine l'arbre orienté augmenté du Naïve Bayes."""

        edges = [f"target->{attr}" for attr in self.graph_V if attr != 'target']
        for attr1, attr2 in self.attr_edges:
            edges.append(f"{attr1}->{attr2}")

        display(utils.drawGraph(";".join(edges)))

#####
# Question 2.4 : comparaison
#####
# Dans notre cas on essaie de prédire la maladie cardiaque, donc les erreurs de type II sont plus importants (faux négatifs) que
# les erreurs de type I (faux positifs). Plus le rappel est faible, plus il y a de faux négatifs.
#
# De même, plus la précision est basse, plus il y a de faux positifs. Un bon classifier ne doit pas diminuer significativement
# le rappel et la précision sur des données autres que celles d'entraînement. L'objectif est donc de minimiser la différence entre le
# rappel et la précision tout en les maximisant sur chaque base de données (train, test).
#
# Nous préférons le classifieur 2D par maximum a posteriori (MAP2DClassifier), car la différence de rappel entre les données
# test et train est minimale (0.004) par rapport aux autres classifieurs. De plus, la précision est presque aussi élevée
# que celle du classifieur ML2DClassifier (0.871 pour 'train', 0.857 pour 'test' contre 0.896 pour 'train', 0.890 pour 'test').
#
# En revanche, le classifieur APrioriClassifier n'est pas le meilleur malgré son rappel parfait (absence des faux négatifs), car
# il produit de nombreuses erreurs de type I, comme le montre sa faible précision (0.745 pour 'train', 0.69 pour 'test').
#####

#####
# Question 3.3.a : Preuve
#####
# Mq P(A,B,C) = P(A) * P(B|A) * P(C|B) avec l'hypothèse : P(A,C|B) = P(A|B) * P(C|B)
# P(A,B,C) = P(A,C|B) * P(B) = P(A|B) * P(C|B) * P(B) = P(A) * P(B|A) * P(C|B) * P(B) / P(B) = P(A) * P(B|A) * P(C|B)
#####

#####
# Question 3.3.b : Complexité en indépendance partielle
#####
# On construit la table de probabilité P pour P(A,B,C):
# 1. Sans l'indépendance conditionnelle -> A,B et C sont dépendants. Il faut stocker les probabilités pour chaque combinaison de valeurs possibles 
# des attributs. Il existe 5^3 combinaisons, donc la taille de la table de probabilité est 5^3 * 8 = 125 * 8 = 1000 octets.
#
# 2. Avec l'indépendance conditionnelle -> e.g. P(A,C|B) = P(A|B) * P(C|B). On a montré que P(A,B,C) = P(A) * P(B|A) * P(C|B).
# Il nous faut stocker P(A), P(B,A), P(C,B), P(B).
# P(A), P(B) : les tableaux 1D -> 5*2 = 10.
# P(B,A), P(C,B) : les tableaux contenant des tuples qui représentent toutes les combinaisons possibles -> 5^2 * 2 = 50.
# La taille de la table de probabilité est (10 + 50) * 8 = 480 octets.
#####

#####
# Question 4.1 : Exemples des graphes
#####
# utils.drawGraphHorizontal("A;B;C;D;E")  # Toutes les variables sont indépendantes : P(A,B,C,D,E) = P(A)P(B)P(C)P(D)P(E)
# utils.drawGraphHorizontal("E->D->C->B->A")  # Il n'existe aucune indépendance : P(A,B,C,D,E) = P(A|BCDE)P(B|CDE)P(C|DE)P(D|E)P(E)
#####

#####
# Question 4.2 : Naïve Bayes
#####
# Hypothèse : les attributs sont indépendants deux à deux conditionnellement à target.
#
# 1. Vraisemblance
# P(attr1, ..., attrN | target) = P(attr1 | target) * ... * P(attrN | target) 
# 
# 2. A posteriori
# P(target | attr1, ..., attrN) = P(attr1, ..., attrN | target) * P(target) / P(attr1, ..., attrN)
# avec P(attr1, ..., attrN) = Sum(target = t) de P(attr1, ..., attrN | target = t) par probabilités totales 
# 
# Pour ne pas calculer cette somme, on pose P(target | attr1, ..., attrN) ~ [proportionnelle] P(attr1, ..., attrN | target) * P(target)
# Dans ce cas, on calcule la partie de droite, nous donnant un vecteur de taille 2 (target = 0 et target = 1). La partie gauche est 
# représentéz aussi par un vecteur de taille 2.
# On a donc P(target | attr1, ..., attrN) = [x1, x2] avec x1 pour target = 0, x2 pour target = 1 et
# P(attr1, ..., attrN | target) * P(target) = [v1, v2] avec v1 pour target = 0, v2 pour target = 1 
# x1 + x2 = 1 par probabilité totale, et comme [x1, x2] est proportionel à [v1,v2], on retrouve facilement [x1, x2] = [v1/(v1+v2), v2/(v1+v2)]
#####

#####
# Question 6.1 : Où se trouve à votre avis le point idéal ?
# Comment pourriez-vous proposer de comparer les différents classifieurs dans cette représentation graphique ?
#####
# Le point idéal est en (1,1), rappel à 1 (absence de faux négatifs) et précision à 1 (absence de faux positifs).
# On peut comparer les classifieurs par rapport à leur position sur les bases train et test et à leur distance avec le point (1,1).
# Plus la position d'un classifieur est proche du point (1,1), meilleure est la qualité de ses prédictions.
# Ainsi, il est également important que la qualité du classifieur ne diminue pas significativement sur la base de données 'test'
# en comparaison avec la base de données 'train'.
#####

#####
# Question 6.3 : Qu'en concluez vous?
#####
# (Observation) entre train et test
# points 6 et 4:
#   précision:  diminue (sur 0.02-0.033)
#   rappel:     diminue (sur 0.54-0.56)
# points  5 et 7
#   précision:  augmente (sur 0.02-0.04)
#   rappel:     diminue (sur 0.5-0.511)
# points  1
#   précision:  diminue (sur 0.055)
#   rappel:     pas de changement
# points  2 et 3
#   précision:  diminue (sur 0.006-0.14)
#   rappel:     augmente (sur 0.004-0.029)
#
# Le rappel est le plus important dans notre cas (voir la Q 2.4 pour plus de détails).
# Il vaut mieux privilégier l'augmentation de rappel qui implique une diminution du nombre de faux négatifs.
# Les classifieurs 6, 4, 5, 7 ont un rappel très bas sur la base de données 'test' et de plus il diminue beaucoup en comparaison avec 'train'.
# Le classifieur 1 a la précision la plus basse des classifieurs.
# Les classifieurs 2, 3, ont les meilleurs caractéristiques : la précision et le rappel sont assez élevés, et le rappel augmente sur 'test'.
# On peut ainsi conclure que les classifieurs 2 et 3 sont meilleurs que les autres au niveau des prédictions, même si sur la base d'entraînement
# ces classifieurs semblaient moins optimaux que les classifieurs 6, 4, 5 et 7.

#####
# Question 8. Quelle leçons & conclusion tirez-vous de ces expériences sur les classifieurs bayésiens ?
#####
# Classifieur Naïve Bayes (sauf TAN) :
# Les classifieurs basés sur ML (maximum vraisemblance) ont la précision plus élevée et le rappel plus faible que les classifieurs
# basées sur MAP (probabilité a posteriori).
# Dans l'ensemble, les classifieurs ont une précision assez élevée (0.89-0.98) pour les deux base de données.
# Cependant, le rappel des résultats de ces classifieurs diminue considérablement sur la base de données test, devenant 2.5 fois plus faible.
#
# Cette dégradation vient de l'hypothèse très forte : tous les attributs sont indépendants conditionnellement à target,
# ce qui n'est pas le cas pour tous les attributs en réalité.
#
# Ainsi, on a réduit les classifeurs MLNaiveBayes et MAPNaiveBayes (ne pas prendre en compte des attributs lors de prédiction s'ils sont
# considérés comme indépendants du target selon le test de khi-deux). Les résultats suivants ont été obtenus :
# Classifieur MAP réduit par rapport à MAPNaiveBayes:
#    - La précision est plus faible sur 'train' et 'test'
#    - Le rappel plus faible sur 'train' et identique sur 'test' -> la réduction de MAPNaiveBayes a amené à la baisse des résultats
# Classifieur ML réduit par rapport à MLNaiveBayes:
#    - La précision plus élevée sur 'train' et 'test'
#    - Le rappel légèrement inférieur (de 0.005) sur 'train' et identique sur 'test' -> la réduction du MLNaiveBayes a permis d'améliorer les prédictions.
#
# Cependant, en raison du rappel très bas sur 'test', on conclut que ces classifieurs ne sont pas adaptés aux prédictions de maladies cardiaques.
#
# Le classifieur TAN dont les prédictions sont basées sur la probabilité a posteriori a la meilleure performance.
# Dans notre implémentation avec le lissage de Laplace avec eps = 1, on reçoit les résultats suivants :
# Train : précision = 0.907, rappel = 0.981
# Test  : précision = 0.876, rappel = 0.92
#
# Les performances du classifieur TAN sont très proches du point idéal (1, 1). 
# On peut conclure que le raffinement du modèle, avec l'évaluation de l'indépendance conditionnelle des attributs par rapport à target à l'aide de l'information mutuelle,
# ainsi qu'au conditionnement des attributs par un autre attribut supplémentaire si possible, permet de mieux représenter les situations réelles et d'améliorer significativement
# les prédictions de classe pour un individu. 
# Plus on autorise des dépendances entre les attributs, plus les probabilités se rapprochent de la situation réelle. Toutefois, cela vient avec une complexité 
# spatiale qui augmente considérablement avec le nombre d'attributs. 
#####
