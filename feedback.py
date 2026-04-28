"""
User feedback collection and interaction logic.
Displays predictions and collects user corrections.
"""

from bylaws import get_bylaw_by_id, get_all_bylaws


def display_prediction(prediction):
    """
    Display the model's prediction to the user.

    Args:
        prediction (dict): Model prediction with bylaw, violated, confidence, reasoning
    """
    bylaw_id = prediction.get("bylaw", "NONE")
    violated = prediction.get("violated", False)
    confidence = prediction.get("confidence", 0.0)
    reasoning = prediction.get("reasoning", "")

    if bylaw_id != "NONE":
        bylaw_info = get_bylaw_by_id(bylaw_id)
        if bylaw_info:
            bylaw_name = bylaw_info["name"]
        else:
            bylaw_name = "Unknown By-law"
    else:
        bylaw_name = "No violation detected"

    print("\n" + "=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)
    print(f"By-law triggered  : {bylaw_id} — {bylaw_name}")
    print(f"Violated          : {'Yes' if violated else 'No'}")
    print(f"Confidence        : {confidence:.2f}")
    print(f"Reasoning         : {reasoning}")
    print("=" * 60)


def collect_user_feedback():
    """
    Collect user feedback on the model's prediction.

    Returns:
        dict: User feedback with correction flags and (optionally) corrected values
    """
    print("\n--- USER FEEDBACK ---")

    # Q1: Is the identified by-law correct?
    while True:
        q1 = input("Q1: Is the identified by-law correct? (y/n): ").strip().lower()
        if q1 in ['y', 'n']:
            bylaw_correct = (q1 == 'y')
            break
        print("  Please enter 'y' or 'n'.")

    # Q2: Is the violation status correct?
    while True:
        q2 = input("Q2: Is the violation status correct? (y/n): ").strip().lower()
        if q2 in ['y', 'n']:
            violation_correct = (q2 == 'y')
            break
        print("  Please enter 'y' or 'n'.")

    feedback = {
        "bylaw_correct": bylaw_correct,
        "violation_correct": violation_correct,
        "corrected_bylaw": None,
        "corrected_violated": None
    }

    # If either is incorrect, collect corrections
    if not bylaw_correct or not violation_correct:
        print("\n--- PROVIDE CORRECTIONS ---")

        if not bylaw_correct:
            print("\nAvailable by-laws:")
            bylaws = get_all_bylaws()
            for bylaw in bylaws:
                print(f"  {bylaw['id']} — {bylaw['name']}")
            print("  NONE — No violation")

            while True:
                corrected_bylaw = input("Enter correct by-law ID (or 'NONE'): ").strip().upper()
                if corrected_bylaw == "NONE" or get_bylaw_by_id(corrected_bylaw):
                    feedback["corrected_bylaw"] = corrected_bylaw
                    break
                print("  Invalid by-law ID. Please try again.")

        if not violation_correct:
            while True:
                corrected_violated = input("Was the by-law violated? (y/n): ").strip().lower()
                if corrected_violated in ['y', 'n']:
                    feedback["corrected_violated"] = (corrected_violated == 'y')
                    break
                print("  Please enter 'y' or 'n'.")

    return feedback


def merge_feedback_with_prediction(prediction, feedback):
    """
    Merge user feedback with model prediction to produce final label.

    Args:
        prediction (dict): Original model prediction
        feedback (dict): User feedback and corrections

    Returns:
        dict: Final label combining prediction and corrections
    """
    final_bylaw = prediction["bylaw"]
    final_violated = prediction["violated"]

    # Apply corrections if provided
    if feedback["corrected_bylaw"] is not None:
        final_bylaw = feedback["corrected_bylaw"]

    if feedback["corrected_violated"] is not None:
        final_violated = feedback["corrected_violated"]

    return {
        "bylaw": final_bylaw,
        "violated": final_violated
    }
