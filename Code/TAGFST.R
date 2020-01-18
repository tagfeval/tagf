library(shiny)
require(class)
library(shiny)

ui <- fluidPage(
  fluidRow(
    column(3, tags$img(height=100, src="logo.jpg")),
    column(9, tags$h1(tags$strong("The Traffic Data Analytics Guidance Framework (TAG-F) Support Tool")))
  ),
  
  tags$hr(),
  tags$br(),
  tags$h2("Overview"),
  tags$hr(),
  #tags$p("TAG-F is a traffic data analytics guidance framework that delineates data-driven traffic prediction as a set of three dimensions: (i) Data Context/Scope (DC), (ii) Data Analytical Method (DAM), and (iii) Data Collection Method (DCM). TAG-F support tool can serve as a decision support mechanism for traffic data scientists by providing guidance in the choice of DAM, given the data context specifications. The framework incorporates seven (7) candidate models ranging from time series, instance-based learning, machine learning, and deep learning models for traffic parameter prediction. The tool provides guidance for traffic data analytics via prediction model suggestion given a set of traffic data parameters. Select the parameters below and click 'Update' when completed."),
  tags$hr(),
  fluidRow(
    column(4, sliderInput(inputId = "sliderPH", label = "Select Prediction Time Steps (minutes)", value = 5, min = 1, max = 120)),
    column(4, selectInput(inputId = "listAL", label = "Select Analysis Level", c("-Select-", "Link", "Junction", "Area"), "-Select-", multiple = FALSE)
    ),
    column(4, selectInput(inputId = "listTrafficScope", label = "Traffic Scope", c("-Select-", "Urban", "Highway/Motorway"), "-Select-", multiple = FALSE))
  ),
  fluidRow(
    column(4, dateRangeInput(inputId = "lblDate", label = "Dataset Date Range", start = NULL, end = NULL, format = "dd/mm/yyyy", separator = "to")),
    column(4, selectInput(inputId = "listGranularity", label = "Traffic Dataset Observation Frequency", c("-Select-", "Daily", "Hourly", "Half-Hourly", "Minutes", "Seconds"), "-Select-", multiple = FALSE)),
    column(4, selectInput(inputId = "listDCM", label = "Traffic Data Collection Method", c("-Select-", "Manual", "ILD", "Bluetooth", "Microwave/Radar", "FCD"), "-Select-", multiple = FALSE))),
  
  fluidRow(
    column(4, checkboxInput(inputId = "cbRealtime", label = "Real-Time Prediction?", value = FALSE)),
    column(4, checkboxInput(inputId = "size", label = "Large Dataset?", value = FALSE)),
    column(4, checkboxInput(inputId = "univariate", label = "Univariate Dataset?", value = FALSE))),
  
  
  
  tags$hr(),
  actionButton(inputId = "go", label = "Update Framework"),
  tags$hr(),
  textOutput(outputId = "txtDataGranularity"),
  tags$hr(),
  textOutput(outputId = "justification"),
  tags$hr(),
  plotOutput("hist")
  
)

server <- function(input, output, session) {
  data <- eventReactive(input$go, {input$sliderPH})
  
  output$hist <- renderPlot({
    model_suggest <- function(analysis, urban, univariate, dcm, large, ph, realtime)
    {
      
      require(ggplot2)
      require(data.table)
      require(randomForest)
      colnames(train)[2] <- "Urban"
      colnames(train)[3] <- "Univariate"
      colnames(train)[4] <- "DCM"
      colnames(train)[5] <- "Large"
      colnames(train)[7] <- "Realtime"
      colnames(train)[8] <- "DAM"
      gdis<-randomForest(DAM ~ ., data=train, ntree=500, keep.forest = TRUE,  importance=TRUE,
                         proximity=TRUE)
      pred <- knn(train = train_x, test = new_data,cl = train_y, k=3)
      summary(pred)
      summary(gdis)
      new_data <- data.frame(Analysis=analysis, Urban=urban, Univariate=univariate, DCM=dcm, Large=large, PH=ph, Realtime=realtime)
      pred.model= predict(gdis,new_data,type="prob")
      new_d <- as.data.frame(t(pred.model))
      library(data.table)
      setDT(new_d, keep.rownames = TRUE)[]
      new_d <- new_d[order(new_d$rn, decreasing=T),]
      colnames(new_d)[1] <- "rn"
      colnames(new_d)[2] <- "V1"
      if ((new_data$Realtime=1)){ reason = "Since Realtime prediction needed, therefore, Suggested DAM is KF"
      } else if ((new_data$Large>=1)&(new_data$PH>=45)){ reason = "Since dataset is large and PH>45, therefore, Suggested PAM is LSTM"
      } else if ((new_data$Large=1)&(new_data$PH<=10)){ reason = "Since dataset is small and PH<10, therefore, Suggested PAM is SVM"
      } else if ((new_data$Large=0)&(new_data$PH>=10)){ reason = "Since dataset is small and PH<10, therefore, Suggested PAM is ARIMA"
      } else if ((new_data$Urban=0)&(new_data$Anlaysis<0)&(new_data$DCM>=1)){reason = "Since traffic scope is non-urban, analysis level is area, DCM not manual, therefore, Suggested PAM is ARIMA"
      } else if ((new_data$Large=0)&(new_data$PH<=1)&(new_data$Realtime<=0)&(new_data$Analysis>=1)){reason = "Since dataset is small, PH is small, and non-realtime prediction required, therefore, Suggested PAM is LR"
      } else if ((new_data$PH>=30)&(new_data$Analysis<=1)){reason = "Since PH is large, and analysis level is link, therefore, Suggested PAM is k-NN"
      } else if ((new_data$PH>=30)&(new_data$Large<=0)&(new_data$Univariate<=0)&(new_data$DCM>=1)){reason = "Since PH is large, and dataset is large multivariate, therefore, Suggested PAM is ANN (also LSTM)"
      } else {
        print("Still thinking")
      }
      return(ggplot(data = new_d, aes(x=reorder(rn, -V1), y=V1)) +
               geom_bar(stat="identity") +
               geom_text(aes(label=round(V1, digits=4)), vjust=1.6, color="white", size=3.5)+
               theme_bw()+ labs(x="Model", y="Probability (Confidence Level)")+
               annotate("label", x = 6, y=0.7, label = reason))
      
    }
    if(input$listDCM =="Manual"){
      data_dcm <- 0
    }
    else if(input$listDCM == "ILD"){
      data_dcm <- 1
    }
    else if(input$listDCM == "Bluetooth")
    {data_dcm <- 2}
    else if(input$listDCM == "Microwave/Radar")
    {data_dcm <- 3}
    else if(input$listDCM == "FCD")
    {data_dcm <- 4}
    
    
    if(input$listAL =="Link"){
      data_al <- 1
    }
    else if(input$listAL == "Area"){
      data_al <- 0
    }
    else if(input$listAL == "Junction")
    {data_al <- 2}
    
    
    if(input$listTrafficScope =="Urban"){
      data_ts <- 1
    }
    else if(input$listTrafficScope == "Highway/Motorway"){
      data_ts <- 0
    }
    
    
    if(input$cbRealtime == FALSE){
      data_rt <- 0
    }
    else {
      data_rt <- 1
    }
    
    
    if(input$size == FALSE){
      data_gran <- 0
    }
    else {
      data_gran <- 1
    }
    
    
    
    if(input$univariate == FALSE){
      univariate <- 0
    }
    else {
      univariate <- 1
    }
    
    if(input$listAL =="Link"){
      data_al <- 1
    }
    else if(input$listAL == "Area"){
      data_al <- 0
    }
    else if(input$listAL == "Junction")
    {data_al <- 2}
    
    
    if(input$listGranularity =="Daily"){
      data_g <- 1
    }
    else if(input$listGranularity == "Hourly"){
      data_g <- 24
    }
    else if(input$listGranularity == "Half-Hourly")
    {data_g <- 48}
    else if(input$listGranularity == "Minutes")
    {data_g <- 48}
    else if(input$listGranularity == "Seconds")
    {data_g <- 48}
    
    
   
    output$txtDataGranularity <- renderText({
      paste("The dataset depth is: ",(data_g*(input$lblDate[2]-input$lblDate[1])))})
    
    output$justification <- renderText({
      paste("Reason: ",(reason))})
    
    model_suggest(data_al,data_ts,1,data_dcm,data_gran,data(),data_rt) 
  })
  
  
  
}

shinyApp(ui, server)