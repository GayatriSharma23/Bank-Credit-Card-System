$(document).ready(function () {
    $("#sendButton").on("click", function () {
        var userInput = $("#data").val();
        if (userInput.trim() !== "") {
            var userMessage = $("<div>").addClass("message user").text(userInput);
            $(".messages").append(userMessage);
            $("#data").val("");

            $.ajax({
                type: "POST",
                url: "/get",
                data: { msg: userInput },
                success: function (response) {
                    var botMessage = $("<div>").addClass("message bot");
                    var botContent = $("<div>").addClass("content");
                    var laptopLogo = $("<img>").attr("src", "laptop.png").attr("alt", "Laptop Logo").css({ width: "20px", height: "20px" });
                    var botText = $("<span>").text(response.answer);

                    botContent.append(laptopLogo).append(botText);
                    botMessage.append(botContent);
                    $(".messages").append(botMessage);

                    if (response.valid) {
                        var graphButton = $("<button>").text("Click to view graph.").addClass("graph-button").on("click", function () {
                            // Logic to display the graph
                            alert("Displaying the graph...");
                        });
                        var sqlButton = $("<button>").text("Show SQL").addClass("sql-button").on("click", function () {
                            alert("SQL Query: " + response.sql_query);
                        });
                        var shareButton = $("<button>").text("Share").addClass("share-button").on("click", function () {
                            // Logic to share the message
                            alert("Sharing the message...");
                        });

                        $(".messages").append(graphButton).append(sqlButton).append(shareButton);
                    }
                },
                error: function () {
                    var botMessage = $("<div>").addClass("message bot").text("Error in processing your request. Please try again.");
                    $(".messages").append(botMessage);
                }
            });
        }
    });
});
