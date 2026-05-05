# ==============================================================================
# TABELLE RULA (dizionario dati)
# ==============================================================================

RULA_TABLE_A = {
    1: {
        1: {1: [1, 2], 2: [2, 2], 3: [2, 3], 4: [3, 3]},
        2: {1: [2, 2], 2: [2, 3], 3: [3, 3], 4: [4, 4]},
        3: {1: [2, 3], 2: [3, 3], 3: [3, 4], 4: [4, 4]}
    },
    2: {
        1: {1: [2, 3], 2: [3, 3], 3: [3, 4], 4: [4, 4]},
        2: {1: [3, 3], 2: [3, 4], 3: [4, 4], 4: [5, 5]},
        3: {1: [3, 4], 2: [4, 4], 3: [4, 5], 4: [5, 5]},
    },
    3: {
        1: {1: [3, 4], 2: [4, 4], 3: [4, 5], 4: [5, 5]},
        2: {1: [4, 4], 2: [4, 5], 3: [5, 5], 4: [6, 6]},
        3: {1: [4, 4], 2: [4, 5], 3: [5, 5], 4: [6, 6]}
    },
    4: {
        1: {1: [4, 4], 2: [4, 5], 3: [5, 5], 4: [6, 6]},
        2: {1: [4, 4], 2: [5, 5], 3: [5, 6], 4: [6, 6]},
        3: {1: [5, 5], 2: [5, 6], 3: [6, 6], 4: [7, 7]}
    },
    5: {
        1: {1: [5, 5], 2: [5, 6], 3: [6, 6], 4: [7, 7]},
        2: {1: [5, 6], 2: [6, 6], 3: [7, 7], 4: [7, 8]},
        3: {1: [6, 6], 2: [6, 7], 3: [7, 7], 4: [8, 8]}
    },
    6: {
        1: {1: [7, 7], 2: [7, 8], 3: [7, 8], 4: [8, 9]},
        2: {1: [8, 8], 2: [8, 8], 3: [8, 9], 4: [9, 9]},
        3: {1: [9, 9], 2: [9, 9], 3: [9, 9], 4: [9, 9]}
    }
}

RULA_TABLE_B = {
    1: {1: [1, 3], 2: [2, 3], 3: [3, 4], 4: [5, 5], 5: [6, 6], 6: [7, 7]},
    2: {1: [2, 3], 2: [2, 3], 3: [4, 5], 4: [5, 5], 5: [6, 7], 6: [7, 7]},
    3: {1: [3, 3], 2: [3, 4], 3: [4, 5], 4: [5, 5], 5: [6, 7], 6: [7, 7]},
    4: {1: [5, 5], 2: [5, 6], 3: [6, 7], 4: [7, 7], 5: [7, 8], 6: [8, 8]},
    5: {1: [7, 7], 2: [7, 7], 3: [7, 8], 4: [8, 8], 5: [8, 8], 6: [8, 9]},
    6: {1: [8, 8], 2: [8, 8], 3: [8, 9], 4: [9, 9], 5: [9, 9], 6: [9, 9]}
}

RULA_FINAL_SCORE = {
    1: {1: 1, 2: 2, 3: 3, 4: 3, 5: 4, 6: 5, 7: 5},
    2: {1: 2, 2: 2, 3: 3, 4: 4, 5: 4, 6: 5, 7: 5},
    3: {1: 3, 2: 3, 3: 3, 4: 4, 5: 5, 6: 5, 7: 6},
    4: {1: 3, 2: 3, 3: 3, 4: 4, 5: 5, 6: 6, 7: 6},
    5: {1: 4, 2: 4, 3: 4, 4: 5, 5: 6, 6: 7, 7: 7},
    6: {1: 4, 2: 4, 3: 5, 4: 6, 5: 6, 6: 7, 7: 7},
    7: {1: 5, 2: 5, 3: 6, 4: 6, 5: 7, 6: 7, 7: 7},
    8: {1: 5, 2: 5, 3: 6, 4: 7, 5: 7, 6: 7, 7: 7}
}

# ==============================================================================
# FUNZIONI DI PUNTEGGIO (Gruppo A: Braccia/Polsi)
# ==============================================================================

def get_upper_arm_score(angle_flex_ext, shoulder_raised=False, across_body=False):
    """Calcola il punteggio RULA per l'Upper Arm (Braccio)"""
    a = abs(angle_flex_ext)
    if a < 20: s = 1
    elif a < 45: s = 2
    elif a < 90: s = 3
    else: s = 4
    
    if shoulder_raised: s += 1
    if across_body: s += 1
    return max(1, min(6, s))

def get_lower_arm_score(angle_flexion):
    """Calcola il punteggio RULA per il Lower Arm (Avambraccio)"""
    a = abs(angle_flexion)
    if 60 <= a <= 100: return 1
    else: return 2

def get_wrist_score(angle_flex_ext, angle_deviation):
    """
    Calcola il punteggio RULA per il Wrist (Polso).

    angle_flex_ext : flessione/estensione in gradi
                     0  = polso neutro   → score 1
                     0-15°               → score 2
                     >15°                → score 3
    angle_deviation: deviazione ulnare/radiale in gradi
                     >5° aggiunge +1

    NOTA: la versione originale usava abs(180-a) che invertiva la scala
    (polso neutro risultava score 3). Questa versione usa direttamente
    il valore assoluto dell angolo di flessione/estensione.
    """
    a = abs(angle_flex_ext)

    if a <= 2:  s = 1          # neutro
    elif a < 15: s = 2         # flessione/estensione lieve
    else:        s = 3         # flessione/estensione marcata

    if abs(angle_deviation) > 5:
        s += 1
    return max(1, min(4, s))

def get_wrist_twist_score(angle_twist):
    """Calcola il punteggio RULA per il Wrist Twist (Torsione Polso)"""
    # NOTA: questo è molto difficile da calcolare da MediaPipe Pose.
    # Stiamo assumendo un valore neutro (1)
    a = abs(angle_twist)
    if  a < 50: s = 1
    else: s= 2
    return max(1, min(2,s))

####va considerato il punteggio maggiore tra left e right
def get_rula_score_a(upper_arm, lower_arm, wrist, wrist_twist):
    """Calcola il Punteggio Finale A dalla Tabella A"""
    try:
        ua = max(1, min(6, upper_arm))
        la = max(1, min(3, lower_arm))
        wr = max(1, min(4, wrist))
        wt = max(1, min(2, wrist_twist))

        step1 = RULA_TABLE_A[ua]
        step2 = step1[la]
        twist_list = step2[wr]
        twist_index = wt - 1
        final_score = twist_list[twist_index]
        return final_score
    except Exception:
        return 0

# ==============================================================================
# FUNZIONI DI PUNTEGGIO (Gruppo B: Collo/Tronco/Gambe)
# ==============================================================================
# NOTA: Queste sono più complesse perché richiedono 3 assi di rotazione (flex, bend, twist)
# che sono difficili da ottenere da una singola webcam.
# Da vedere come trattare questi casi in futuro.

###Provare a togliere qualche grado 
def get_neck_score(angle_flex_ext, angle_lateral_bend, angle_twist):
    """Calcola il punteggio RULA per il Neck (Collo)"""
    a = angle_flex_ext 
    if 0 <= a <= 10: s = 1
    elif a <= 20: s = 2
    elif a > 20: s = 3
    elif a < 0: s = 4 # Estensione
    
    if (abs(angle_lateral_bend) > 5): s += 1
    if (abs(angle_twist) > 5): s += 1
    return max(1, min(6, s))

def get_trunk_score(angle_flexion, angle_lateral_bend, angle_twist):
    """Calcola il punteggio RULA per il Trunk (Tronco)"""
    a = abs(angle_flexion)
    if a < 5: s = 1
    elif a < 20: s = 2
    elif a < 60: s = 3
    else: s = 4
    
    if (abs(angle_lateral_bend) > 5): s += 1
    if (abs(angle_twist) > 5): s += 1
    return max(1, min(6, s))

def get_leg_score(seated=True, balanced=True):
    """Calcola il punteggio RULA per le Legs (Gambe)"""
    if seated and balanced: return 1
    else: return 2

def get_rula_score_b(neck_score, trunk_score, legs_score):
    """Calcola il Punteggio Finale B dalla Tabella B"""
    try:
        nk = max(1, min(6, neck_score))
        tr = max(1, min(6, trunk_score))
        lg = max(1, min(2, legs_score))

        step1 = RULA_TABLE_B[nk]
        legs_list = step1[tr]
        legs_index = lg - 1
        final_score = legs_list[legs_index]
        return final_score
    except Exception:
        return 0

# ==============================================================================
# FUNZIONE DI PUNTEGGIO (Gruppo C: Finale)
# ==============================================================================

def get_rula_score_c(score_A, score_B, muscle_use=False, force_load=False):
    """Calcola il Punteggio Finale C dalla Tabella C"""
    try:
        A = min(score_A, 8)
        B = min(score_B, 7)
        
        score_C = RULA_FINAL_SCORE[A][B]
        
        if muscle_use:
            score_C += 1
        if force_load:
            score_C += 1
        return score_C
    except Exception:
        return 0