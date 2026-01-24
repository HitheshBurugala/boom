# Take input from user
a = float(input("Enter first number: "))
b = float(input("Enter second number: "))

# Basic operations
addition = a + b
subtraction = a - b
multiplication = a * b

# Handle division safely
division = a / b if b != 0 else "Undefined (division by zero)"
floor_division = a // b if b != 0 else "Undefined (division by zero)"
modulo = a % b if b != 0 else "Undefined (division by zero)"

# Other math operations
power = a ** b
absolute_difference = abs(a - b)
maximum = max(a, b)
minimum = min(a, b)

# Display results
print("\n--- Results ---")
print("Addition:", addition)
print("Subtraction:", subtraction)
print("Multiplication:", multiplication)
print("Division:", division)
print("Floor Division:", floor_division)
print("Modulo:", modulo)
print("Power (a^b):", power)
print("Absolute Difference:", absolute_difference)
print("Maximum:", maximum)
print("Minimum:", minimum)
