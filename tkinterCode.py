import tkinter as tk

#define function to make the window and display results
def makeWindow():
    # Create the root window
    root = tk.Tk()
    root.geometry("500x500")

    # Create a frame to hold the labels
    frame = tk.Frame(root)

    # Create two labels with the text we want to display
    label1 = tk.Label(frame, text="Line 1", font=("Arial", 18), justify="center", pady=50)
    label2 = tk.Label(frame, text="Line 2", font=("Arial", 18), justify="center", pady=50)

    # Pack the labels into the frame, with some padding
    label1.pack()
    label2.pack()

    # Pack the frame into the root window, centering it horizontally and vertically
    frame.pack(fill="both", expand=True)
    root.after(5000, root.destroy)
    
    # Start the main event loop
    root.mainloop()

makeWindow()