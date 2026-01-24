import streamlit as st
import math

# Page config
st.set_page_config(page_title="Simple Calculator", layout="centered")

st.title("🧮 Simple Calculator (2 Numbers)")
st.write("Enter two numbers and see all math operations instantly.")

# Input fields
num1 = st.number_input("Enter first number", value=0.0)
num2 = st.number_input("Enter second number", value=0.0)

st.divider()

# Calculations
addition = num1 + num2
subtraction = num1 - num2
multiplication = num1 * num2

division = "Undefined (division by zero)" if num2 == 0 else num1 / num2
floor_division = "Undefined (division by zero)" if num2 == 0 else num1 // num2
modulo = "Undefined (division by zero)" if num2 == 0 else num1 % num2

power = num1 ** num2
absolute_difference = abs(num1 - num2)
maximum = max(num1, num2)
minimum = min(num1, num2)

# Display results
st.subheader("Results")

st.write(f"➕ Addition: **{addition}**")
st.write(f"➖ Subtraction: **{subtraction}**")
st.write(f"✖️ Multiplication: **{multiplication}**")
st.write(f"➗ Division: **{division}**")
st.write(f"📐 Floor Division: **{floor_division}**")
st.write(f"🔁 Modulo: **{modulo}**")
st.write(f"⬆️ Power (a^b): **{power}**")
st.write(f"📏 Absolute Difference: **{absolute_difference}**")
st.write(f"🔼 Maximum: **{maximum}**")
st.write(f"🔽 Minimum: **{minimum}**")
