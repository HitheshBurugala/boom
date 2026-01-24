import streamlit as st
import math

# Page config
st.set_page_config(page_title="Simple Calculator", layout="centered")

st.title("Simple Calculator (2 Numbers)")
st.write("Enter two numbers. All operations are exception-safe.")

# Input fields
num1 = st.number_input("Enter first number", value=0.0)
num2 = st.number_input("Enter second number", value=0.0)

st.divider()

# Helper function to safely execute operations
def safe_operation(func, error_message="Invalid operation"):
    try:
        return func()
    except ZeroDivisionError:
        return "Error: Division by zero"
    except OverflowError:
        return "Error: Number too large"
    except ValueError:
        return "Error: Invalid value"
    except Exception:
        return error_message

# Calculations (exception-safe)
addition = safe_operation(lambda: num1 + num2)
subtraction = safe_operation(lambda: num1 - num2)
multiplication = safe_operation(lambda: num1 * num2)

division = safe_operation(lambda: num1 / num2)
floor_division = safe_operation(lambda: num1 // num2)
modulo = safe_operation(lambda: num1 % num2)

power = safe_operation(lambda: num1 ** num2)
absolute_difference = safe_operation(lambda: abs(num1 - num2))
maximum = safe_operation(lambda: max(num1, num2))
minimum = safe_operation(lambda: min(num1, num2))

# Display results
st.subheader("Results")

st.write(f"Addition: **{addition}**")
st.write(f"Subtraction: **{subtraction}**")
st.write(f"Multiplication: **{multiplication}**")
st.write(f"Division: **{division}**")
st.write(f"Floor Division: **{floor_division}**")
st.write(f"Modulo: **{modulo}**")
st.write(f"Power (a^b): **{power}**")
st.write(f"Absolute Difference: **{absolute_difference}**")
st.write(f"Maximum: **{maximum}**")
st.write(f"Minimum: **{minimum}**")
