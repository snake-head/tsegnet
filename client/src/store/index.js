// 引入
import { createStore } from "vuex";


export default createStore({
  // 声明变量
  state: {
    currentUrl: null,
    vtkContext: null,
    ifDisplayCentroids: false,
    centroidsPred: [],
    sceneData: [],
    modelColor: [],
    ifDisplayResult: false,
  },
  // mutations的值由actions传入
  actions: {
    setCurrentUrl(context, value){
      context.commit('SetCurrentUrl', value)
    },
    setVtkContext(context, value){
      context.commit('SetVtkContext', value)
    },
    deleteVtkContext(context){
        context.commit('DeleteVtkContext')
    },
    deleteCentroidsPred(context){
      context.commit('DeleteCentroidsPred')
    },
    setCentroidsPred(context, value){
      context.commit('SetCentroidsPred', value)
    },
    displayCentroids(context, value){
      context.commit('DisplayCentroids', value)
    },
    setSceneData(context, value){
      context.commit('SetSceneData', value)
    },
    setModelColor(context, value){
      context.commit('SetModelColor', value)
    },
    displayResult(context, value){
      context.commit('DisplayResult', value)
    },
},
  // 修改变量（state不能直接赋值修改，只能通过mutations）
  mutations: {
    SetCurrentUrl(state, value){
      state.currentUrl = value
    },
    SetVtkContext(state, value){
      state.vtkContext = value
    },
    DeleteVtkContext(state){
      const { genericRenderer, actor, mapper } = state.vtkContext;
      actor.delete();
      mapper.delete();
      genericRenderer.delete();
      state.vtkContext = null
    },
    DeleteCentroidsPred(state){
      state.centroidsPred = []
    },
    SetCentroidsPred(state, value){
      state.centroidsPred = value
    },
    DisplayCentroids(state, value){
      state.ifDisplayCentroids = value
    },
    SetSceneData(state, value){
      state.sceneData = value
    },
    SetModelColor(state, value){
      state.modelColor = value
    },
    DisplayResult(state, value){
      state.ifDisplayResult = value
    },
  },
  modules: {},
});

