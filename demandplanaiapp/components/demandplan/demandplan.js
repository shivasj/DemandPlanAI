(function() {
    'use strict';
    angular.module('demandplanai.app.demandplan', [])
        .config(function ($stateProvider, $urlRouterProvider) {
            var now = new Date();
            var ticks = now.getTime();

            // Demand Plan
            $stateProvider.state('demandplan', {
                url: '/demandplan',
                templateUrl: 'components/demandplan/demandplan.html?'+ticks,
                controller: 'DemandPlanController',
                controllerAs: 'demandPlanCtrl'
            });

        })
        .controller('DemandPlanController',function(DataService,AppService,$state, $stateParams,$timeout){
            var self = this;

            // Build demand table for next 10 days
            self.demand_days = []
            self.today = (new Date('4/27/2018')).getTime();
            var tomorrow = self.today + 1 * 24 * 60 * 60 * 1000;
            self.demand_days.push(tomorrow);
            for(var i=1;i<10;i++){
                tomorrow = self.demand_days[i-1];
                self.demand_days.push(self.demand_days[i-1] + 1 * 24 * 60 * 60 * 1000)
            }
            console.log(self.demand_days);

            // Get Commodities
            DataService.GetCommodity()
                .then(function(response){
                    self.commodity = response.data.slice(0, -1);

                    // Get Demand History
                    return DataService.GetDemandHistory();
                })
                .then(function(response) {
                    self.demand_history = response.data.slice(0, -1);

                    // Load top models
                    return DataService.GetTopModels();
                })
                .then(function(response) {
                    self.top_models = response.data.slice(0, -1);

                    // InitPage
                    self.InitPage();
                })
                .catch(function(response){
                    console.dir(response);
                    //toaster.pop('error', "Error", "There was an error processing your request");
                });

            self.demandchart = [];
            self.demanddiv = [];
            self.selected_model = [];
            self.total_predictions = [0,0,0,0,0,0,0,0,0,0]
            self.InitPage = function () {
                _.forEach(self.commodity, function(d) {
                    // Get demand data for commodity
                    var data = _.filter(self.demand_history, { 'commodity': d.commodity });
                    console.log(data)

                    // History Chart
                    self.demandchart[d.commodity] = d3components.demandChart()
                        .margin({ top: 0, right: 0, bottom: 0, left: 40 })
                        .responsive(true)
                        .displayXAxis(true)
                        .height(100)
                        .fixedHeight(false);
                    self.demanddiv[d.commodity] = d3.select("#demandHistory"+d.commodity);
                    self.demanddiv[d.commodity].datum({history:data,prediction:[]}).call(self.demandchart[d.commodity]);

                    // Select top model
                    self.SelectTopModel(d.commodity);
                });

                //self.InitPredictions();
                $timeout(self.InitPredictions(), 5000);
            }
            self.SelectTopModel = function (commodity) {
                // Selected Model
                var top_model = _.filter(self.top_models, { 'Type': commodity })[0];
                self.selected_model[commodity] = top_model;

            }
            self.PreparePredictionData = function (data) {
                var results = [
                    {date:'4/28/2018',qty:0},{date:'4/29/2018',qty:0},{date:'4/30/2018',qty:0},{date:'5/1/2018',qty:0},{date:'5/2/2018',qty:0}
                    ,{date:'5/3/2018',qty:0},{date:'5/4/2018',qty:0},{date:'5/5/2018',qty:0},{date:'5/6/2018',qty:0},{date:'5/7/2018',qty:0}
                ];
                var tomorrow = self.today + 1 * 24 * 60 * 60 * 1000;
                _.forEach(data,function (d,i) {
                    // results.push({
                    //     date:tomorrow,qty: +d[0]
                    // })
                    results[i]['qty'] = +d[0]
                    self.total_predictions[i] = +d[0];

                    tomorrow = tomorrow + 1 * 24 * 60 * 60 * 1000;
                });
                return results;
            }
            self.InitPredictions = function () {
                // Predict

                var top_model = self.selected_model['Nut'];
                // Predict
                DataService.Predict(top_model['key'],top_model['epochs'],top_model['look_back'],'Nut')
                    .then(function(response){
                        //self.commodity = response.data.slice(0, -1);
                        console.log(response.data);

                        var data = _.filter(self.demand_history, { 'commodity': 'Nut' });
                        var predictions = self.PreparePredictionData(response.data.future_pred_actuals);
                        self.demanddiv['Nut'].datum({history:data,prediction:predictions}).call(self.demandchart['Nut']);


                        var top_model = self.selected_model['Supplement'];
                        return DataService.Predict(top_model['key'],top_model['epochs'],top_model['look_back'],'Supplement');
                    })
                    .then(function(response){
                        //self.commodity = response.data.slice(0, -1);
                        console.log(response.data);

                        var data = _.filter(self.demand_history, { 'commodity': 'Supplement' });
                        var predictions = self.PreparePredictionData(response.data.future_pred_actuals);
                        self.demanddiv['Supplement'].datum({history:data,prediction:predictions}).call(self.demandchart['Supplement']);


                        var top_model = self.selected_model['BreadCrumb'];
                        return DataService.Predict(top_model['key'],top_model['epochs'],top_model['look_back'],'BreadCrumb');
                    })
                    .then(function(response){
                        //self.commodity = response.data.slice(0, -1);
                        console.log(response.data);

                        var data = _.filter(self.demand_history, { 'commodity': 'BreadCrumb' });
                        var predictions = self.PreparePredictionData(response.data.future_pred_actuals);
                        self.demanddiv['BreadCrumb'].datum({history:data,prediction:predictions}).call(self.demandchart['BreadCrumb']);
                    })
                    .catch(function(response){
                        console.dir(response);
                        //toaster.pop('error', "Error", "There was an error processing your request");
                    });
            }
            self.PredictForModel = function (data,top_model) {
                // Predict
                DataService.Predict(top_model['key'],top_model['epochs'],top_model['look_back'])
                    .then(function(response){
                        //self.commodity = response.data.slice(0, -1);
                        console.log(response.data)
                    })
                    .catch(function(response){
                        console.dir(response);
                        //toaster.pop('error', "Error", "There was an error processing your request");
                    });
            }
            self.GetTopModels = function (comm) {
                return _.filter(self.top_models, { 'Type': comm });
            }
            self.PredictForSelectedModel = function (commodity) {
                // Predict
                var top_model = self.selected_model[commodity];
                // Predict
                DataService.Predict(top_model['key'],top_model['epochs'],top_model['look_back'],commodity)
                    .then(function(response){
                        //self.commodity = response.data.slice(0, -1);
                        console.log(response.data);

                        var data = _.filter(self.demand_history, { 'commodity': commodity });
                        var predictions = self.PreparePredictionData(response.data.future_pred_actuals);
                        self.demanddiv[commodity].datum({history:data,prediction:predictions}).call(self.demandchart[commodity]);
                    })
                    .catch(function(response){
                        console.dir(response);
                        //toaster.pop('error', "Error", "There was an error processing your request");
                    });
            }

        })

})();